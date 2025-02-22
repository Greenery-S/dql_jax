import jax,flax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
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

# -------------------- Diffusion 模型 -------------------- #
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
        # 定义noise模型
        self.model = ActionPredictorMLP(self.state_dim, self.action_dim)
        # 定义损失函数
        if self.loss_type == "l2":
            self.loss_fn = lambda pred, target, weights: jnp.mean(
                ((pred - target) ** 2) * weights
            )
        elif self.loss_type == "l1":
            self.loss_fn = lambda pred, target, weights: jnp.mean(
                jnp.abs(pred - target) * weights
            )
        else:
            raise NotImplementedError(f"loss type {self.loss_type} not implemented")

        # 选择 beta 安排
        if self.beta_schedule == "linear":
            betas = linear_beta_schedule(self.n_timesteps)
        elif self.beta_schedule == "cosine":
            betas = cosine_beta_schedule(self.n_timesteps)
        elif self.beta_schedule == "vp":
            betas = vp_beta_schedule(self.n_timesteps)
        else:
            raise ValueError(f"unkown beta_schedule: {self.beta_schedule}")

        # 计算 alpha 与相关量
        alphas = 1.0 - betas
        alphas_cumprod = jnp.cumprod(alphas, axis=0)
        alphas_cumprod_prev = jnp.concatenate(
            [jnp.ones((1,), dtype=alphas_cumprod.dtype), alphas_cumprod[:-1]], axis=0
        )
        alphas_cumprod = alphas_cumprod
        alphas_cumprod_prev = alphas_cumprod_prev
        sqrt_alphas_cumprod = jnp.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - alphas_cumprod)
        log_one_minus_alphas_cumprod = jnp.log(1.0 - alphas_cumprod)
        sqrt_recip_alphas_cumprod = jnp.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = jnp.sqrt(1.0 / alphas_cumprod - 1.0)

        # 计算后验 q(x_{t-1} | x_t, x0) 相关量
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        posterior_log_variance_clipped = jnp.log(
            jnp.clip(posterior_variance, a_min=1e-20)
        )
        posterior_mean_coef1 = (
            betas * jnp.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * jnp.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

        # 临时保存相关量
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = sqrt_alphas_cumprod
        self.sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod
        self.log_one_minus_alphas_cumprod = log_one_minus_alphas_cumprod
        self.sqrt_recip_alphas_cumprod = sqrt_recip_alphas_cumprod
        self.sqrt_recipm1_alphas_cumprod = sqrt_recipm1_alphas_cumprod
        self.posterior_variance = posterior_variance
        self.posterior_log_variance_clipped = posterior_log_variance_clipped
        self.posterior_mean_coef1 = posterior_mean_coef1
        self.posterior_mean_coef2 = posterior_mean_coef2

    # ---------- 采样(推理)部分 ---------- #
    def predict_start_from_noise(
        self, x_t: jnp.ndarray, t: jnp.ndarray, noise: jnp.ndarray
    ) -> jnp.ndarray:
        """
        根据 x_t 与预测 noise 还原出 x0。
        """
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start: jnp.ndarray, x_t: jnp.ndarray, t: jnp.ndarray):
        """
        计算后验分布 q(x_{t-1} | x_t, x0) 的均值与对数方差。
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x: jnp.ndarray, t: jnp.ndarray, state: jnp.ndarray):
        """
        根据当前 x_t 计算模型预测的均值与对数方差。
        """
        noise_pred = self.model(x, t, state)
        x_recon = self.predict_start_from_noise(x, t, noise_pred)
        if self.clip_denoised:
            x_recon = jnp.clip(x_recon, -self.max_action, self.max_action)
        model_mean, model_var, model_log_variance = self.q_posterior(x_recon, x, t)
        return model_mean, model_var, model_log_variance

    def p_sample(
        self, x: jnp.ndarray, t: jnp.ndarray, state: jnp.ndarray, rng: jnp.ndarray
    ) -> jnp.ndarray:
        """
        在当前时间步 t 下采样 x_{t-1}。
        """
        model_mean, _, model_log_variance = self.p_mean_variance(x, t, state)
        rng, subkey = jax.random.split(rng)
        noise = jax.random.normal(subkey, shape=x.shape)
        nonzero_mask = (t != 0).astype(x.dtype)
        nonzero_mask = nonzero_mask.reshape(
            (nonzero_mask.shape[0],) + (1,) * (len(x.shape) - 1)
        )
        return model_mean + nonzero_mask * jnp.exp(0.5 * model_log_variance) * noise

    # TODO: 可能需要修改为 lax control_flow, 有点难度
    def p_sample_loop(
        self,
        state: jnp.ndarray,
        shape: tuple,
        rng: jnp.ndarray,
        verbose: bool = False,
        return_diffusion: bool = False,
    ) -> Any:
        """
        从纯噪声开始，反向采样直到得到样本。
        """
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
        """
        给定条件 state 采样 action。
        """
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        rng, subkey = jax.random.split(rng)
        action = self.p_sample_loop(state, shape, subkey, **kwargs)
        return jnp.clip(action, -self.max_action, self.max_action)

    # ---------- 训练部分 ---------- #
    def q_sample(
        self,
        x_start: jnp.ndarray,
        t: jnp.ndarray,
        rng: jnp.ndarray,
        noise: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        根据 x0 采样 x_t。
        """
        if noise is None:
            noise = jax.random.normal(rng, shape=x_start.shape)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(
        self,
        x_start: jnp.ndarray,
        state: jnp.ndarray,
        t: jnp.ndarray,
        rng: jnp.ndarray,
        weights: float = 1.0,
    ) -> jnp.ndarray:
        """
        计算训练损失。
        """
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

    def loss(
        self, x: jnp.ndarray, state: jnp.ndarray, rng: jnp.ndarray, weights: float = 1.0
    ) -> jnp.ndarray:
        """
        随机采样时间步 t 并计算损失。
        """
        batch_size = x.shape[0]
        rng, subkey = jax.random.split(rng)
        t = jax.random.randint(
            subkey, shape=(batch_size,), minval=0, maxval=self.n_timesteps
        )
        return self.p_losses(x, state, t, rng, weights)

    def __call__(self, state: jnp.ndarray, rng: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """
        默认调用时返回采样结果。
        """
        return self.sample(state, rng, **kwargs)


# -------------------- MLP 模型 -------------------- #
class ActionPredictorMLP(nn.Module):
    """
    mlp map action,timestep,state to action.
    f(a,t,s) -> a
    t need embedding.
    """

    state_dim: int
    action_dim: int
    t_dim: int = 16

    def setup(self):
        self.time_emb = nn.Sequential(
            [
                SinusoidalPosEmb(self.t_dim),
                nn.Dense(self.t_dim * 2),
                mish,
                nn.Dense(self.t_dim),
            ]
        )
        self.net = nn.Sequential(
            [
                nn.Dense(256),
                mish,
                nn.Dense(256),
                mish,
                nn.Dense(256),
                mish,
                nn.Dense(self.action_dim),
            ]
        )

    def __call__(
        self, x: jnp.ndarray, t: jnp.ndarray, state: jnp.ndarray
    ) -> jnp.ndarray:
        t_emb = self.time_emb(t)
        x = jnp.concatenate([x, t_emb, state], axis=-1)
        return self.net(x)
