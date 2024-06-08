import gymnasium as gym
import envs.catanEnv
import numpy as np
import os
import pathlib
import argparse
import matplotlib.pyplot as plt

# imports for PPO
from stable_baselines3.common.env_checker import check_env
from sb3_contrib.common.maskable.policies  import MaskableMultiInputActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.monitor import Monitor

def mask_fn(env: gym.Env) -> np.ndarray:
    valid_actions = env.get_legal_actions()
    mask = np.zeros(env.action_space.n, dtype=np.float32)
    mask[valid_actions] = 1
    return np.array([bool(i) for i in mask])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rl-steps",
        type=int,
        help="The number of learning iterations",
        default=10000
    )
    parser.add_argument(
        "--iterations",
        type=int,
        help='Number of times to run training',
        default = 100
    )
    parser.add_argument(
        "--update-interval",
        type=int,
        default=10
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=1000
    )
    args = parser.parse_args()

    output_path = pathlib.Path(__file__).parent.joinpath(
        "results",
        f"Catan-ppo",
    )
    log_path = f'./results/Catan-ppo/logs/'

    env = gym.make('Catan-v0')
    env = ActionMasker(env, mask_fn)  # Wrap to enable masking
    #check_env(env, warn=True)
    model = MaskablePPO(MaskableMultiInputActorCriticPolicy, env, verbose=1)
    
    eval_env = gym.make('Catan-v0')
    eval_env = Monitor(eval_env)
    eval_env = ActionMasker(eval_env, mask_fn)
    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path='./results/Catan-ppo/model/',
        log_path=log_path,
        eval_freq=args.eval_freq,
        render=False
    )

    # Training the model
    if not output_path.exists():
        output_path.mkdir(parents=True)
    for iteration in range(args.iterations):
        print("ITERATION: ", str(iteration))
        model.learn(total_timesteps=args.rl_steps, callback=eval_callback, progress_bar=True)

        if iteration % args.update_interval == 0:
            current_policy_path = f'./results/current_policy_{iteration}.pth'
            model.save(current_policy_path)
            # update env policy to newly saved policy
            env.update_fixed_policy(model.load(current_policy_path))
    

    final_model_path = output_path.joinpath("final_model.zip")
    model.save(final_model_path)

    # Load the evaluations.npz file
    evaluation_file = './results/Catan-ppo/logs/evaluations.npz'
    evaluations = np.load(evaluation_file)

    # Extract the data
    timesteps = evaluations['timesteps']
    results = evaluations['results']

    # Calculate mean and standard deviation of rewards
    mean_rewards = np.mean(results, axis=1)
    std_rewards = np.std(results, axis=1)

    # Plot the mean rewards with error bars for standard deviation
    plt.figure(figsize=(10, 6))
    plt.errorbar(timesteps, mean_rewards, yerr=std_rewards, label='Mean Reward +/- Std Dev')
    plt.xlabel('Timesteps')
    plt.ylabel('Mean Reward')
    plt.title('Evaluation Mean Reward over Time')
    plt.legend()
    plt.grid()
    plt.show()