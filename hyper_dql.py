import ml_collections

def get_default_config():
    config = ml_collections.ConfigDict(
        {
            "actor_lr": 3e-4,
            "critic_lr": 3e-4,
            "hidden_dims": (256, 256, 256),
            "discount": 0.99,
            "target_update_rate": 5e-3,
            "ema_update_rate": 5e-3,
            "warmup_steps": 1000,
            "ema_update_interval": 5,
            "eta": 1.0,
            "is_max_q_backup": False,
            "max_q_repeat": 10,
            "beta_schedule": "vp",
            "loss_type": "l2",
            "n_timesteps": 5,
            "opt_max_steps": 2e6,
            "is_critic_opt_decay": False,
            "is_actor_opt_decay": False,
            "grad_norm_clip": 2.0,
            "top_k": 1,
            "reward_tune": "no",
            "num_samples": 50,
            "temperature": 1.0,
        }
    )
    return config

hyperparameters = {
    'halfcheetah-medium-v2':         {'actor_lr': 3e-4,'critic_lr':3e-4, 'eta': 1.0,   'is_max_q_backup': False,  'reward_tune': 'no',          'grad_norm_clip': 9.0,  'top_k': 1},
    'hopper-medium-v2':              {'actor_lr': 3e-4,'critic_lr':3e-4, 'eta': 1.0,   'is_max_q_backup': False,  'reward_tune': 'no',          'grad_norm_clip': 9.0,  'top_k': 2},
    'walker2d-medium-v2':            {'actor_lr': 3e-4,'critic_lr':3e-4, 'eta': 1.0,   'is_max_q_backup': False,  'reward_tune': 'no',          'grad_norm_clip': 1.0,  'top_k': 1},
    'halfcheetah-medium-replay-v2':  {'actor_lr': 3e-4,'critic_lr':3e-4, 'eta': 1.0,   'is_max_q_backup': False,  'reward_tune': 'no',          'grad_norm_clip': 2.0,  'top_k': 0},
    'hopper-medium-replay-v2':       {'actor_lr': 3e-4,'critic_lr':3e-4, 'eta': 1.0,   'is_max_q_backup': False,  'reward_tune': 'no',          'grad_norm_clip': 4.0,  'top_k': 2},
    'walker2d-medium-replay-v2':     {'actor_lr': 3e-4,'critic_lr':3e-4, 'eta': 1.0,   'is_max_q_backup': False,  'reward_tune': 'no',          'grad_norm_clip': 4.0,  'top_k': 1},
    'halfcheetah-medium-expert-v2':  {'actor_lr': 3e-4,'critic_lr':3e-4, 'eta': 1.0,   'is_max_q_backup': False,  'reward_tune': 'no',          'grad_norm_clip': 7.0,  'top_k': 0},
    'hopper-medium-expert-v2':       {'actor_lr': 3e-4,'critic_lr':3e-4, 'eta': 1.0,   'is_max_q_backup': False,  'reward_tune': 'no',          'grad_norm_clip': 5.0,  'top_k': 2},
    'walker2d-medium-expert-v2':     {'actor_lr': 3e-4,'critic_lr':3e-4, 'eta': 1.0,   'is_max_q_backup': False,  'reward_tune': 'no',          'grad_norm_clip': 5.0,  'top_k': 1},
    'antmaze-umaze-v0':              {'actor_lr': 3e-4,'critic_lr':3e-4, 'eta': 0.5,   'is_max_q_backup': False,  'reward_tune': 'cql_antmaze', 'grad_norm_clip': 2.0,  'top_k': 2},
    'antmaze-umaze-diverse-v0':      {'actor_lr': 3e-4,'critic_lr':3e-4, 'eta': 2.0,   'is_max_q_backup': True,   'reward_tune': 'cql_antmaze', 'grad_norm_clip': 3.0,  'top_k': 2},
    'antmaze-medium-play-v0':        {'actor_lr': 1e-3,'critic_lr':1e-3, 'eta': 2.0,   'is_max_q_backup': True,   'reward_tune': 'cql_antmaze', 'grad_norm_clip': 2.0,  'top_k': 1},
    'antmaze-medium-diverse-v0':     {'actor_lr': 3e-4,'critic_lr':3e-4, 'eta': 3.0,   'is_max_q_backup': True,   'reward_tune': 'cql_antmaze', 'grad_norm_clip': 1.0,  'top_k': 1},
    'antmaze-large-play-v0':         {'actor_lr': 3e-4,'critic_lr':3e-4, 'eta': 4.5,   'is_max_q_backup': True,   'reward_tune': 'cql_antmaze', 'grad_norm_clip': 10.0, 'top_k': 2},
    'antmaze-large-diverse-v0':      {'actor_lr': 3e-4,'critic_lr':3e-4, 'eta': 3.5,   'is_max_q_backup': True,   'reward_tune': 'cql_antmaze', 'grad_norm_clip': 7.0,  'top_k': 1},
    'pen-human-v1':                  {'actor_lr': 3e-5,'critic_lr':3e-5, 'eta': 0.15,  'is_max_q_backup': False,  'reward_tune': 'normalize',   'grad_norm_clip': 7.0,  'top_k': 2},
    'pen-cloned-v1':                 {'actor_lr': 3e-5,'critic_lr':3e-5, 'eta': 0.1,   'is_max_q_backup': False,  'reward_tune': 'normalize',   'grad_norm_clip': 8.0,  'top_k': 2},
    'kitchen-complete-v0':           {'actor_lr': 3e-4,'critic_lr':3e-4, 'eta': 0.005, 'is_max_q_backup': False,  'reward_tune': 'no',          'grad_norm_clip': 9.0,  'top_k': 2},
    'kitchen-partial-v0':            {'actor_lr': 3e-4,'critic_lr':3e-4, 'eta': 0.005, 'is_max_q_backup': False,  'reward_tune': 'no',          'grad_norm_clip': 10.0, 'top_k': 2},
    'kitchen-mixed-v0':              {'actor_lr': 3e-4,'critic_lr':3e-4, 'eta': 0.005, 'is_max_q_backup': False,  'reward_tune': 'no',          'grad_norm_clip': 10.0, 'top_k': 0},
}