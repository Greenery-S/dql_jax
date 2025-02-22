import jax
import jax.numpy as jnp
from flax import linen as nn

##############################################
# network & model utils
##############################################

def mish(x):
    return x * jnp.tanh(nn.softplus(x))

class SinusoidalPosEmb(nn.Module):
    dim: int

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: 输入张量，形状通常为 (batch,) 或 (batch, 1)，每个元素代表一个位置或时间步
        Returns:
            输出张量形状为 (batch, dim)，由 sin 与 cos 编码拼接而成
        """
        half_dim = self.dim // 2
        # 使用 jnp.log 计算缩放因子（与 torch 中 math.log 保持一致）
        scale = jnp.log(10000.0) / (half_dim - 1)
        # 生成一个从 0 到 half_dim-1 的序列，并计算指数
        emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -scale)
        # x[:, None] 扩展维度，与 emb[None, :] 相乘，广播后得到形状 (batch, half_dim)
        emb = x[:, None] * emb[None, :]
        # 计算 sin 与 cos 并拼接到最后一维
        return jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)


def extract(a: jnp.ndarray, t: jnp.ndarray, x_shape: tuple) -> jnp.ndarray:
    """
    模仿 PyTorch 的实现：从数组 a 的最后一个维度上，根据索引张量 t 提取对应元素，
    并 reshape 成形状 (b, 1, 1, ..., 1)，其中 b = t.shape[0]，后面的 1 的数量为 len(x_shape) - 1。
    
    参数：
      a: jnp.ndarray，通常是一个预先计算好的系数序列，形状至少为 (T,) 或 (..., T)
      t: jnp.ndarray，包含索引，形状 (b, …)（通常为 (b,)）
      x_shape: 用于确定需要扩展的维度数量（例如目标张量的形状）
    
    返回：
      形状为 (b, 1, 1, ..., 1) 的数组，可以与形状为 x_shape 的数组进行广播
    """
    # 取出批量大小 b
    b, *_ = t.shape
    # 使用 jnp.take 沿 a 的最后一个维度提取 t 对应的元素，
    # 这里 axis=-1 与 PyTorch 中的 a.gather(-1, t) 对应
    out = jnp.take(a, t, axis=-1)
    # reshape 到 (b, 1, 1, ..., 1)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

##############################################
# ddpm beta schedules
##############################################

def cosine_beta_schedule(timesteps: int, s: float = 0.008, dtype=jnp.float32) -> jnp.ndarray:
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = jnp.linspace(0, steps, steps)
    alphas_cumprod = jnp.cos(((x / steps) + s) / (1 + s) * jnp.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = jnp.clip(betas, a_min=0, a_max=0.999)
    return betas_clipped.astype(dtype)


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2, dtype=jnp.float32) -> jnp.ndarray:
    betas = jnp.linspace(beta_start, beta_end, timesteps)
    return betas.astype(dtype)


def vp_beta_schedule(timesteps: int, dtype=jnp.float32) -> jnp.ndarray:
    t = jnp.arange(1, timesteps + 1)
    T = timesteps
    b_max = 10.0
    b_min = 0.1
    alpha = jnp.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / (T ** 2))
    betas = 1 - alpha
    return betas.astype(dtype)


##############################################
# reward tuning
##############################################

def get_iql_normalization(dataset):
        returns = []
        ret = 0
        for r, term in zip(dataset['rewards'], dataset['dones_float']):
            ret += r
            if term:
                returns.append(ret)
                ret = 0
        return (max(returns) - min(returns)) / 1000
    
def get_tuned_dataset(dataset, reward_tune):
    if reward_tune == 'no':
        pass
    elif reward_tune == 'iql_locomotion':
        normalizing_factor = get_iql_normalization(dataset)
        dataset = dataset.copy({'rewards': dataset['rewards'] / normalizing_factor})
    elif reward_tune == 'normalize':
        mean = dataset['rewards'].mean()
        std = dataset['rewards'].std()
        dataset = dataset.copy({'rewards': (dataset['rewards'] - mean) / std})
    elif reward_tune == 'cql_antmaze':
        dataset = dataset.copy({'rewards': (dataset['rewards']-0.5 )*4.0})
    elif reward_tune == 'iql_antmaze':
        dataset = dataset.copy({'rewards': dataset['rewards']-1.0})
    elif reward_tune == 'antmaze':
        dataset = dataset.copy({'rewards': (dataset['rewards']-0.25)*2.0})
    return dataset

def main():
    # 测试SinusoidalPosEmb

    # 测试数据
    x = jnp.arange(10)  # [0, 1, 2, ..., 9]
    emb = SinusoidalPosEmb(dim=16)
    result = emb(x)
    print("x:", x)
    print("result:", result.shape)

if __name__ == '__main__':
    main()