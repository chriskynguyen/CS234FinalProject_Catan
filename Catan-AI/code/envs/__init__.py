from gym.envs.registration import register
from code.envs.catanEnv import CatanEnv

register(
    id='Catan-v0',
    entry_point='code.envs.catanEnv:CatanEnv',
    max_episode_steps=1000,
)