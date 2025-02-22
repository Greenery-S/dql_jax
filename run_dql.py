import os
from absl import app, flags
from functools import partial
from numpy.random import choice

from jax import random
import tqdm
import algo_dql as learner  # a local self defined module
from util_dql import get_tuned_dataset
from hyper_dql import hyperparameters, get_default_config
import jaxrl_m.examples.mujoco.d4rl_utils as d4rl_utils  # a local self defined / reused module

from jaxrl_m.wandb import setup_wandb, default_wandb_config, get_flag_dict
import wandb
from jaxrl_m.evaluation import supply_rng, evaluate

from ml_collections import config_flags
import pickle
from flax.training import checkpoints


FLAGS = flags.FLAGS
flags.DEFINE_string("env_name", "hopper-medium-v2", "Environment name.")

flags.DEFINE_string("save_dir", None, "Logging dir (if not None, save params).")

flags.DEFINE_integer("seed", choice(1000000), "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 50000, "Eval interval.")
flags.DEFINE_integer("save_interval", 50000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(2e6), "Number of training steps.")

wandb_config = default_wandb_config()
wandb_config.update(
    {
        "project": "d4rl_test",
        "group": "dql_test",
        "name": "dql_{env_name}",
    }
)

config_flags.DEFINE_config_dict("wandb", wandb_config, lock_config=False)
config_flags.DEFINE_config_dict("config", get_default_config(), lock_config=False)

overlay = {
    # "actor_lr": 3e-4,
    # "critic_lr": 3e-4,
    # "hidden_dims": (256, 256, 256),
    # "discount": 0.99,
    # "target_update_rate": 5e-3,
    # "ema_update_rate": 5e-3,
    # "warmup_steps": 1000,
    # "ema_update_interval": 5,
    # "eta": 1.0,
    "is_max_q_backup": True,
    "max_q_repeat": 3,
    # "beta_schedule": "vp",
    # "loss_type": "l2",
    # "n_timesteps": 5,
    # "opt_max_steps":1e6,
    # "is_actor_opt_decay": True,
    # "is_critic_opt_decay": True,
    # "grad_norm_clip": 1.0,
    # "top_k": 1,
    # "reward_tune": "no",
    # "num_samples": 200,
    # "temperature": 0.1,
}


def main(_):

    env_name = FLAGS.env_name
    env_cfg = hyperparameters.get(env_name, {})
    FLAGS.config.update(env_cfg)
    FLAGS.config.update(overlay)

    # Create wandb logger
    setup_wandb(FLAGS.config.to_dict(), **FLAGS.wandb)

    if FLAGS.save_dir is not None:
        FLAGS.save_dir = os.path.join(
            FLAGS.save_dir,
            wandb.run.project,
            wandb.config.exp_prefix,
            wandb.config.experiment_id,
        )
        os.makedirs(FLAGS.save_dir, exist_ok=True)
        print(f"Saving config to {FLAGS.save_dir}/config.pkl")
        with open(os.path.join(FLAGS.save_dir, "config.pkl"), "wb") as f:
            pickle.dump(get_flag_dict(), f)

    env = d4rl_utils.make_env(env_name)
    dataset = d4rl_utils.get_dataset(env)
    reward_tune = FLAGS.config.reward_tune
    dataset = get_tuned_dataset(dataset, reward_tune)

    example_batch = dataset.sample(1)
    agent = learner.create_learner(
        FLAGS.seed,
        example_batch["observations"],
        example_batch["actions"],
        max_action=env.action_space.high[0],
        # opt_max_steps=FLAGS.max_steps,
        **FLAGS.config,
    )
    temperature = FLAGS.config.get("temperature", 1.0)

    for i in tqdm.tqdm(
        range(1, FLAGS.max_steps + 1), smoothing=0.1, dynamic_ncols=True
    ):

        batch = dataset.sample(FLAGS.batch_size)
        agent, update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            train_metrics = {f"training/{k}": v for k, v in update_info.items()}
            wandb.log(train_metrics, step=i)

        if i % FLAGS.eval_interval == 0:
            policy_fn = partial(
                supply_rng(
                    agent.sample_actions,
                    rng=random.PRNGKey(choice(1000000)),
                ),
                temperature=temperature,
            )
            eval_info = evaluate(policy_fn, env, num_episodes=FLAGS.eval_episodes)

            eval_metrics = {f"evaluation/{k}": v for k, v in eval_info.items()}
            wandb.log(eval_metrics, step=i)

        if i % FLAGS.save_interval == 0 and FLAGS.save_dir is not None:
            checkpoints.save_checkpoint(FLAGS.save_dir, agent, i)


if __name__ == "__main__":
    app.run(main)
