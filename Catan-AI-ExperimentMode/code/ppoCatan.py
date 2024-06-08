import gym
import catanEnv
import numpy as np

# imports for PPO
from stable_baselines3.common.env_checker import check_env
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

def mask_fn(env: gym.Env) -> np.ndarray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.valid_action_mask()


env = gym.make('Catan-v0')
env = ActionMasker(env, mask_fn)  # Wrap to enable masking

model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1)
model.learn()