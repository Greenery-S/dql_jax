import jax
import jax.numpy as jnp
import optax
import flax
from flax import linen as nn
from jaxrl_m.common import target_update, TrainState
from jaxrl_m.typing import PRNGKey, Batch, InfoDict
from model_dql import *
from jax.lax import stop_gradient
from jax.random import uniform
from jaxrl_m.networks import Policy, ValueCritic, Critic, ensemblize
from typing import Sequence, Optional
from pprint import pprint
import distrax


class DQLAgent(flax.struct.PyTreeNode):
    rng: PRNGKey
    actor: TrainState
    ema_actor: TrainState  # 和target_critic的更新方式一样
    critic: TrainState
    target_critic: TrainState
    config: dict = flax.struct.field(pytree_node=False)
    # 是否是树节点, 类似于torch的register_buffer
    step: int = 0

    @jax.jit
    def update(agent, batch: Batch) -> InfoDict:
        def critic_loss_fn(critic_params):
            new_rng, subkey = jax.random.split(agent.rng)
            q1, q2 = agent.critic(
                batch["observations"], batch["actions"], params=critic_params
            )
            if agent.config["is_max_q_backup"]: #TODO: 这个分支有问题, 就算repeats==1也不和不使用时等效
                repeats = agent.config["max_q_repeat"]
                next_state_rpt = jnp.repeat(batch["next_observations"], repeats, axis=0)
                next_action_rpt = agent.ema_actor(next_state_rpt, subkey)
                q1_nxt, q2_nxt = agent.target_critic(next_state_rpt, next_action_rpt)
                q1_nxt = q1_nxt.reshape(-1, repeats)
                q2_nxt = q2_nxt.reshape(-1, repeats)
                # maxmin
                # candidate_q = jnp.minimum(q1_nxt, q2_nxt)
                # q_nxt = jnp.max(candidate_q, axis=1, keepdims=True)
                # minmax
                q1_nxt_max = jnp.max(q1_nxt, axis=1, keepdims=True)
                q2_nxt_max = jnp.max(q2_nxt, axis=1, keepdims=True)
                q_nxt = jnp.minimum(q1_nxt_max, q2_nxt_max).reshape(-1)
            else:
                action_nxt = agent.ema_actor(batch["next_observations"], subkey)
                q1_nxt, q2_nxt = agent.target_critic(
                    batch["next_observations"], action_nxt
                )
                q_nxt = jnp.minimum(q1_nxt, q2_nxt)

            target_q = stop_gradient(
                batch["rewards"] + agent.config["discount"] * batch["masks"] * q_nxt
            )
            critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
            return critic_loss, {
                "critic_loss": critic_loss,
                "q1_mean": q1.mean(),
                "q2_mean": q2.mean(),
                "q1_std": q1.std(),
                "q2_std": q2.std(),
                "new_rng": new_rng,
            }

        def actor_loss_fn(actor_params):
            new_rng, *subkey = jax.random.split(agent.rng, 4)
            bc_loss = agent.actor(
                batch["actions"],
                batch["observations"],
                subkey[0],
                params=actor_params,
                method=Diffusion.loss,
            )
            action_nxt = agent.actor(batch["observations"], subkey[1])
            q1_nxt, q2_nxt = agent.critic(batch["observations"], action_nxt)
            q_loss = jnp.where(
                uniform(subkey[2]) < 0.5,
                -q1_nxt.mean() / stop_gradient(jnp.abs(q2_nxt).mean()),
                -q2_nxt.mean() / stop_gradient(jnp.abs(q1_nxt).mean()),
            )
            actor_loss = bc_loss + q_loss * agent.config["eta"]
            return actor_loss, {
                "actor_loss": actor_loss,
                "bc_loss": bc_loss,
                "q_loss": q_loss,
                "new_rng": new_rng,
            }

        new_critic, critic_info = agent.critic.apply_loss_fn(
            loss_fn=critic_loss_fn, has_aux=True
        )
        agent = agent.replace(rng=critic_info["new_rng"])
        new_actor, actor_info = agent.actor.apply_loss_fn(
            loss_fn=actor_loss_fn, has_aux=True
        )
        agent = agent.replace(rng=actor_info["new_rng"])

        new_target_critic = target_update(
            agent.critic, agent.target_critic, agent.config["target_update_rate"]
        )

        condition = jnp.logical_and(
            agent.step >= agent.config["warmup_steps"],
            jnp.equal(jnp.mod(agent.step + 1, agent.config["ema_update_interval"]), 0),
        )
        new_ema_actor = jax.lax.cond(
            condition,
            lambda _: target_update(
                agent.actor, agent.ema_actor, agent.config["ema_update_rate"]
            ),
            lambda _: agent.ema_actor,
            operand=None,
        )

        return agent.replace(
            critic=new_critic,
            target_critic=new_target_critic,
            actor=new_actor,
            ema_actor=new_ema_actor,
            step=agent.step + 1,
        ), {**critic_info, **actor_info}

    @jax.jit
    def sample_actions(
        agent,
        obs: jnp.ndarray,
        *,
        seed: PRNGKey,
        temperature: float = 1.0,
        # 用于控制 softmax 的温度, 越小越接近 argmax
    ) -> jnp.ndarray:
        # 重复 obs
        obs = jnp.asarray(obs)[None, ...]
        obs_rpt = jnp.repeat(obs, repeats=agent.config["num_samples"], axis=0)

        # 批量采样候选动作
        action_rpt = agent.actor(obs_rpt, rng=seed, method=Diffusion.sample)

        # 计算 q 值
        q1, q2 = agent.target_critic(obs_rpt, action_rpt)
        q_min = jnp.minimum(q1, q2).reshape(-1)

        # 用 distrax 的 Categorical 分布，从 logits 做一次采样
        # distrax.Categorical(logits=...)会等效于 softmax(...)
        dist = distrax.Categorical(logits=q_min / temperature)
        seed, subkey = jax.random.split(seed)
        idx = dist.sample(seed=subkey)  # 返回 [0..49] 间的一个索引

        return action_rpt[idx]


def create_learner(
    seed: int,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    max_action: float = 1.0,
    actor_lr: float = 3e-4,
    critic_lr: float = 3e-4,
    hidden_dims: Sequence[int] = (256, 256, 256),
    discount: float = 0.99,
    target_update_rate: float = 5e-3,
    ema_update_rate: float = 5e-3,
    warmup_steps: int = 5000,
    ema_update_interval: int = 5,
    eta: float = 1.0,
    is_max_q_backup: bool = False,
    max_q_repeat: int = 10,
    beta_schedule: str = "vp",
    loss_type: str = "l2",
    n_timesteps: int = 100,
    is_critic_opt_decay: bool = False,
    is_actor_opt_decay: bool = False,
    opt_max_steps: int = 2e6,
    grad_norm_clip: float = 5.0,
    num_samples: int = 50,
    **kwargs,
):
    key = jax.random.PRNGKey(seed)
    action_dim = actions.shape[-1]
    state_dim = observations.shape[-1]
    diffusion_def = Diffusion(
        state_dim,
        action_dim,
        max_action=max_action,
        beta_schedule=beta_schedule,
        n_timesteps=n_timesteps,
        loss_type=loss_type,
    )

    ##############################
    # DIFFUSION ACTOR
    ##############################
    if is_actor_opt_decay:
        schedule_fn = optax.cosine_decay_schedule(-actor_lr, opt_max_steps)
        actor_tx = optax.chain(
            optax.clip_by_global_norm(grad_norm_clip),
            optax.scale_by_adam(),
            optax.scale_by_schedule(schedule_fn),
        )
    else:
        actor_tx = optax.chain(
            optax.clip_by_global_norm(grad_norm_clip),
            optax.adam(learning_rate=actor_lr),
        )
    key, *subkey = jax.random.split(key, 3)
    (out, actor_var) = diffusion_def.init_with_output(
        subkey[0],
        observations,
        subkey[1],
    )
    actor_params = actor_var["params"]

    pprint(jax.tree.map(lambda x: x.shape, actor_params))
    pprint(out)
    diffusion_actor = TrainState.create(
        diffusion_def,
        tx=actor_tx,
        params=actor_params,
    )
    ema_actor = TrainState.create(diffusion_def, params=actor_params)

    ##############################
    # DOUBLE Q CRITIC
    ##############################
    key, subkey = jax.random.split(key)
    critic_def = ensemblize(Critic, num_qs=2)(hidden_dims, mish)
    if is_critic_opt_decay:
        schedule_fn = optax.cosine_decay_schedule(-critic_lr, opt_max_steps)
        critic_tx = optax.chain(
            optax.clip_by_global_norm(grad_norm_clip),
            optax.scale_by_adam(),
            optax.scale_by_schedule(schedule_fn),
        )
    else:
        critic_tx = optax.chain(
            optax.clip_by_global_norm(grad_norm_clip),
            optax.adam(learning_rate=critic_lr),
        )
    critic_params = critic_def.init(subkey, observations, actions)["params"]
    critic = TrainState.create(
        critic_def,
        critic_params,
        tx=critic_tx,
    )
    target_critic = TrainState.create(critic_def, critic_params)

    config = flax.core.FrozenDict(
        dict(
            discount=discount,
            target_update_rate=target_update_rate,
            ema_update_rate=ema_update_rate,
            warmup_steps=warmup_steps,
            ema_update_interval=ema_update_interval,
            eta=eta,
            is_max_q_backup=is_max_q_backup,
            max_q_repeat=max_q_repeat,
            num_samples=num_samples,
        )
    )

    return DQLAgent(
        rng=key,
        actor=diffusion_actor,
        ema_actor=ema_actor,
        critic=critic,
        target_critic=target_critic,
        config=config,
    )
