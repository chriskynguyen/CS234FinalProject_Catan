from gymnasium.envs.registration import register
from envs.catanEnv import CatanEnv

register(
    id='Catan-v0',
    entry_point='envs.catanEnv:CatanEnv',
    max_episode_steps=1000,
)