import jax
import optax
from model_dql import *
from jaxrl_m.typing import *
from jaxrl_m.common import TrainState, target_update
from jaxrl_m.wandb import setup_wandb, default_wandb_config, get_flag_dict
import jaxrl_m.examples.mujoco.d4rl_utils as d4rl_utils  # a local self defined / reused module
import wandb
from pprint import pprint
from absl import app, flags


def main(_):
    # 数据
    env_name = "halfcheetah-expert-v2"
    env = d4rl_utils.make_env(env_name)
    dataset = d4rl_utils.get_dataset(env)

    # Create wandb logger
    wandb_config = {
        "project": "d4rl_test",
        "group": "dql_test",
        "name": f"dql_ddpm_{env_name}",
    }
    setup_wandb(default_wandb_config(), **wandb_config)

    if False:
        for k in dataset:
            print(k, jax.tree.map(lambda x: x.shape, dataset[k]))

    # 模型
    one_sample = dataset.sample(1)
    for k in one_sample:
        print(k, jax.tree.map(lambda x: x.shape, one_sample[k]))
    action_dim = one_sample["actions"].shape[-1]
    state_dim = one_sample["observations"].shape[-1]
    diffusion_def = Diffusion(state_dim, action_dim, max_action=1.0)

    # 训练
    key = jax.random.PRNGKey(0)
    key, *subkey = jax.random.split(key, 3)
    (out, params) = diffusion_def.init_with_output(
        subkey[0],
        one_sample["observations"],
        subkey[1],
    )

    pprint(jax.tree.map(lambda x: x.shape, params))
    pprint(out)

    params = params["params"]

    diffusion_model = TrainState.create(
        diffusion_def,
        tx=optax.adamw(3e-4),
        params=params,
    )

    @jax.jit # jit处谨慎使用函数闭包...
    def update(diffusion_model:TrainState,batch: Batch, key: PRNGKey) -> InfoDict:
        key, subkey = jax.random.split(key)
        def loss_fn(params):
            loss = diffusion_model(
                batch["actions"],
                batch["observations"],
                rng=subkey,
                params=params,
                method=Diffusion.loss,
            )
            return loss, {"training loss": loss}

        new_diffusion_model, info = diffusion_model.apply_loss_fn(
            loss_fn=loss_fn, has_aux=True
        )
        return new_diffusion_model, info

    eval_table = wandb.Table(
        columns=["step", "type"] + [f"action_dim_{i}" for i in range(action_dim)],
    )

    # 模拟训练
    import tqdm

    for i in tqdm.trange(100000):
        batch = dataset.sample(256)
        key, subkey = jax.random.split(key)
        diffusion_model, info = update(diffusion_model,batch, subkey)

        if (i + 1) % 5000 == 0:
            wandb.log(info, step=i + 1)
            key, subkey = jax.random.split(key)
            test_data = dataset.sample(1)
            action = diffusion_model(test_data["observations"], subkey)
            eval_table.add_data(i + 1, "gt", *test_data["actions"][0].tolist())
            eval_table.add_data(i + 1, "pred", *action[0].tolist())

    wandb.log({"eval": eval_table})


if __name__ == "__main__":
    app.run(main)
