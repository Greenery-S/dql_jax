import os
import pickle
from absl import app, flags
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state, checkpoints
from ml_collections import config_flags
import tqdm
import wandb

# 从你项目内的模块导入 wandb 相关函数
from jaxrl_m.wandb import setup_wandb, default_wandb_config, get_flag_dict
import jaxrl_m.examples.mujoco.d4rl_utils as d4rl_utils  # 本地自定义/复用的模块

# -------------------- Flags & Config -------------------- #
FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'halfcheetah-expert-v2', 'Environment name.')
flags.DEFINE_string('save_dir', None, 'Logging dir (if not None, save params).')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('log_interval', 50, 'Logging interval.')
flags.DEFINE_integer('max_steps', 1000, 'Number of training steps.')
flags.DEFINE_integer('batch_size', 256, 'Batch size.')

# wandb 配置（使用默认配置并更新部分信息）
wandb_config = default_wandb_config()
wandb_config.update({
    'project': 'd4rl_test',
    'group': 'iql_test',
    'name': f'diffusion_{FLAGS.env_name}',
})
config_flags.DEFINE_config_dict('wandb', wandb_config, lock_config=False)

# -------------------- Diffusion 模型相关 -------------------- #
from typing import Any, Callable, Optional
from functools import partial
from util_dql import (
    extract,
    linear_beta_schedule,
    cosine_beta_schedule,
    vp_beta_schedule,
    SinusoidalPosEmb,
    mish,
)

class ActionPredictorMLP(nn.Module):
    """
    用于预测噪声的 MLP。
    输入：x, timestep embedding, state；输出：预测噪声或直接预测 x0
    """
    state_dim: int
    action_dim: int
    t_dim: int = 16

    def setup(self):
        self.time_emb = nn.Sequential([
            SinusoidalPosEmb(self.t_dim),
            nn.Dense(self.t_dim * 2),
            mish,
            nn.Dense(self.t_dim),
        ])
        self.net = nn.Sequential([
            nn.Dense(256),
            mish,
            nn.Dense(256),
            mish,
            nn.Dense(self.action_dim),
        ])

    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, state: jnp.ndarray) -> jnp.ndarray:
        t_emb = self.time_emb(t)
        x = jnp.concatenate([x, t_emb, state], axis=-1)
        return self.net(x)

class Diffusion(nn.Module):
    state_dim: int
    action_dim: int
    max_action: float
    beta_schedule: str = "vp"  # 可选 'linear', 'cosine', 'vp'
    n_timesteps: int = 100
    loss_type: str = "l2"  # 目前支持 'l2' 和 'l1'
    clip_denoised: bool = True
    predict_epsilon: bool = True  # True 时 model 输出 noise，否则输出 x0

    def setup(self):
        self.model = ActionPredictorMLP(self.state_dim, self.action_dim)
        # 选择 beta 安排
        if self.beta_schedule == "linear":
            betas = linear_beta_schedule(self.n_timesteps)
        elif self.beta_schedule == "cosine":
            betas = cosine_beta_schedule(self.n_timesteps)
        elif self.beta_schedule == "vp":
            betas = vp_beta_schedule(self.n_timesteps)
        else:
            raise ValueError(f"unknown beta_schedule: {self.beta_schedule}")
        self.betas = betas
        alphas = 1.0 - betas
        alphas_cumprod = jnp.cumprod(alphas, axis=0)
        alphas_cumprod_prev = jnp.concatenate(
            [jnp.ones((1,), dtype=alphas_cumprod.dtype), alphas_cumprod[:-1]], axis=0
        )

        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        self.sqrt_alphas_cumprod = jnp.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - alphas_cumprod)
        self.log_one_minus_alphas_cumprod = jnp.log(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = jnp.sqrt(1.0 / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = jnp.sqrt(1.0 / alphas_cumprod - 1.0)

        # 计算后验 q(x_{t-1} | x_t, x0) 相关量
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_variance = posterior_variance
        self.posterior_log_variance_clipped = jnp.log(jnp.clip(posterior_variance, a_min=1e-20))
        self.posterior_mean_coef1 = betas * jnp.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * jnp.sqrt(alphas) / (1.0 - alphas_cumprod)

        # 定义损失函数
        if self.loss_type == "l2":
            self.loss_fn: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray] = (
                lambda pred, target, weights: jnp.mean(((pred - target) ** 2) * weights)
            )
        elif self.loss_type == "l1":
            self.loss_fn: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray] = (
                lambda pred, target, weights: jnp.mean(jnp.abs(pred - target) * weights)
            )
        else:
            raise NotImplementedError(f"loss type {self.loss_type} not implemented")

    def predict_start_from_noise(self, x_t: jnp.ndarray, t: jnp.ndarray, noise: jnp.ndarray) -> jnp.ndarray:
        if self.predict_epsilon:
            return extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - \
                   extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        else:
            return noise

    def q_posterior(self, x_start: jnp.ndarray, x_t: jnp.ndarray, t: jnp.ndarray):
        posterior_mean = (extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                          extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x: jnp.ndarray, t: jnp.ndarray, state: jnp.ndarray):
        noise_pred = self.model(x, t, state)
        x_recon = self.predict_start_from_noise(x, t, noise_pred)
        if self.clip_denoised:
            x_recon = jnp.clip(x_recon, -self.max_action, self.max_action)
        model_mean, model_var, model_log_variance = self.q_posterior(x_recon, x, t)
        return model_mean, model_var, model_log_variance

    def p_sample(self, x: jnp.ndarray, t: jnp.ndarray, state: jnp.ndarray, rng: jnp.ndarray) -> jnp.ndarray:
        model_mean, _, model_log_variance = self.p_mean_variance(x, t, state)
        rng, subkey = jax.random.split(rng)
        noise = jax.random.normal(subkey, shape=x.shape)
        nonzero_mask = (t != 0).astype(x.dtype)
        nonzero_mask = nonzero_mask.reshape((nonzero_mask.shape[0],) + (1,) * (len(x.shape) - 1))
        return model_mean + nonzero_mask * jnp.exp(0.5 * model_log_variance) * noise

    def p_sample_loop(self, state: jnp.ndarray, shape: tuple, rng: jnp.ndarray,
                      verbose: bool = False, return_diffusion: bool = False) -> Any:
        rng, subkey = jax.random.split(rng)
        x = jax.random.normal(subkey, shape)
        diffusion = [x] if return_diffusion else None

        for i in reversed(range(self.n_timesteps)):
            t = jnp.full((shape[0],), i, dtype=jnp.int32)
            rng, subkey = jax.random.split(rng)
            x = self.p_sample(x, t, state, subkey)
            if verbose:
                print(f"t = {i}")
            if return_diffusion:
                diffusion.append(x)
        if return_diffusion:
            diffusion = jnp.stack(diffusion, axis=1)
            return x, diffusion
        else:
            return x

    def sample(self, state: jnp.ndarray, rng: jnp.ndarray, **kwargs) -> jnp.ndarray:
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        rng, subkey = jax.random.split(rng)
        action = self.p_sample_loop(state, shape, subkey, **kwargs)
        return jnp.clip(action, -self.max_action, self.max_action)

    def q_sample(self, x_start: jnp.ndarray, t: jnp.ndarray, rng: jnp.ndarray,
                 noise: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        if noise is None:
            noise = jax.random.normal(rng, shape=x_start.shape)
        return (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def p_losses(self, x_start: jnp.ndarray, state: jnp.ndarray, t: jnp.ndarray, rng: jnp.ndarray,
                 weights: float = 1.0) -> jnp.ndarray:
        rng, subkey = jax.random.split(rng)
        noise = jax.random.normal(subkey, shape=x_start.shape)
        rng, subkey = jax.random.split(rng)
        x_noisy = self.q_sample(x_start, t, subkey, noise)
        x_recon = self.model(x_noisy, t, state)
        if self.predict_epsilon:
            loss = self.loss_fn(x_recon, noise, weights)
        else:
            loss = self.loss_fn(x_recon, x_start, weights)
        return loss

    def loss(self, x: jnp.ndarray, state: jnp.ndarray, rng: jnp.ndarray,
             weights: float = 1.0) -> jnp.ndarray:
        batch_size = x.shape[0]
        rng, subkey = jax.random.split(rng)
        t = jax.random.randint(subkey, shape=(batch_size,), minval=0, maxval=self.n_timesteps)
        return self.p_losses(x, state, t, rng, weights)

    def __call__(self, state: jnp.ndarray, rng: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return self.sample(state, rng, **kwargs)

def diffusion_p_sample_loop(params: Any, state: jnp.ndarray, shape: tuple, rng: jnp.ndarray,
                             n_timesteps: int, diffusion_def: Diffusion) -> tuple:
    rng, subkey = jax.random.split(rng)
    x = jax.random.normal(subkey, shape)

    def body_fn(carry, i):
        rng, x = carry
        t = jnp.full((shape[0],), n_timesteps - 1 - i, dtype=jnp.int32)
        rng, subkey = jax.random.split(rng)
        x = diffusion_def.apply(params, x, t, state, rng=subkey, method=Diffusion.p_sample)
        return (rng, x), x

    (rng, x), xs = jax.lax.scan(body_fn, (rng, x), jnp.arange(n_timesteps))
    return x, xs

# -------------------- 训练部分 -------------------- #
# 我们这里用 jaxrl_m.typing 中定义的类型做类型注解（如果没有可以忽略）
from jaxrl_m.typing import Batch, PRNGKey, InfoDict

# 定义 TrainState，方便管理参数和优化器状态
# 若项目中已有 TrainState 定义，可直接使用
class TrainState(train_state.TrainState):
    model: Any = None  # 可选，记录模型定义

@partial(jax.jit, static_argnums=(3,))
def train_step(state: TrainState, batch: dict, rng: jnp.ndarray, diffusion: Diffusion):
    def loss_fn(params):
        loss = diffusion.apply(params, batch["actions"], batch["observations"],
                               rng=rng, method=Diffusion.loss)
        return loss
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss

def main(_):
    # -------------------- wandb 初始化与配置保存 -------------------- #
    setup_wandb(FLAGS.config.to_dict(), **FLAGS.wandb)

    if FLAGS.save_dir is not None:
        save_path = os.path.join(FLAGS.save_dir, wandb.run.project,
                                 wandb.config.get("exp_prefix", "default"), wandb.run.id)
        os.makedirs(save_path, exist_ok=True)
        print(f'Saving config to {save_path}/config.pkl')
        with open(os.path.join(save_path, 'config.pkl'), 'wb') as f:
            pickle.dump(get_flag_dict(), f)

    # -------------------- 数据加载 -------------------- #
    env = d4rl_utils.make_env(FLAGS.env_name)
    dataset = d4rl_utils.get_dataset(env)

    # 打印 sample 数据的 shape
    one_sample = dataset.sample(1)
    for k in one_sample:
        print(k, jax.tree_map(lambda x: x.shape, one_sample[k]))

    action_dim = one_sample["actions"].shape[-1]
    state_dim = one_sample["observations"].shape[-1]

    # -------------------- 模型初始化 -------------------- #
    diffusion = Diffusion(state_dim, action_dim, max_action=1.0)
    key = jax.random.PRNGKey(FLAGS.seed)
    key, subkey, subkey2 = jax.random.split(key, 3)
    # 使用 init_with_output 初始化模型（同时得到输出和参数）
    out, params = diffusion.init_with_output(subkey, one_sample["observations"], subkey2)
    print("Initial parameter shapes:")
    print(jax.tree_map(lambda x: x.shape, params))

    diffusion_state = TrainState.create(
        apply_fn=diffusion.apply,
        params=params,
        tx=optax.adam(1e-3),
        model=diffusion,
    )

    # -------------------- 模拟训练循环 -------------------- #
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1), dynamic_ncols=True, smoothing=0.1):
        batch = dataset.sample(FLAGS.batch_size)
        key, subkey = jax.random.split(key)
        diffusion_state, loss = train_step(diffusion_state, batch, subkey, diffusion)

        if i % FLAGS.log_interval == 0:
            wandb.log({"training_loss": loss}, step=i)
            key, subkey = jax.random.split(key)
            test_data = dataset.sample(1)
            action = diffusion.apply(diffusion_state.params, test_data["observations"], subkey)
            # 这里简单记录 action 的数值（转换为 list 便于 wandb 展示）
            wandb.log({"sampled_action": action.tolist()}, step=i)
            print("Sampled action:", action)
            print("True action:", test_data["actions"])

        if FLAGS.save_dir is not None and i % FLAGS.log_interval == 0:
            checkpoint_path = os.path.join(FLAGS.save_dir, 'checkpoints')
            os.makedirs(checkpoint_path, exist_ok=True)
            checkpoints.save_checkpoint(checkpoint_path, diffusion_state, i)

if __name__ == "__main__":
    app.run(main)
