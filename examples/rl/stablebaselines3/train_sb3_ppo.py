import gym
import gym_multirotor
from stable_baselines3 import PPO, A2C, SAC
from sb3_contrib import TRPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

import wandb
from wandb.integration.sb3 import WandbCallback


SEED = 123
ENV_NAME = "QuadrotorPlusHoverEnv-v0"
num_envs = 1
total_timesteps = 5000000
model_name = "trpo"

use_wandb = True
COMMIT = "different noise level"

env = gym.make(ENV_NAME)  # ENV_NAMES = ["QuadrotorXHoverEnv-v0", "TiltrotorPlus8DofHoverEnv-v0", "QuadrotorPlusHoverEnv-v0"]
noise_level = "0" + str(env.observation_noise_std)[2:]

if model_name == "ppo":
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=f'logs/log_{ENV_NAME}',
        policy_kwargs=dict(activation_fn=torch.nn.ReLU, net_arch=dict(pi=[256, 256], vf=[256, 256])),
        learning_rate=0.00005,
        clip_range=0.05,
        seed=SEED,
        batch_size=256,
        max_grad_norm=0.2
    )
elif model_name == "a2c":
    model = A2C(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=f'logs/log_{ENV_NAME}',
        policy_kwargs=dict(activation_fn=torch.nn.ReLU, net_arch=dict(pi=[256, 256], vf=[256, 256])),
        learning_rate=0.00005,
        seed=SEED,
        max_grad_norm=0.2
    )
elif model_name == "sac":
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=f'logs/log_{ENV_NAME}',
        policy_kwargs=dict(activation_fn=torch.nn.ReLU, net_arch=dict(pi=[256, 256], qf=[256, 256])),
        learning_rate=0.00005,
        buffer_size=int(1e6),
        batch_size=256,
        seed=SEED,
    )
elif model_name == "trpo":
    model = TRPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=f'logs/log_{ENV_NAME}',
        policy_kwargs=dict(activation_fn=torch.nn.ReLU, net_arch=dict(pi=[256, 256], vf=[256, 256])),
        learning_rate=0.00005,
        seed=SEED,
        batch_size=256
    )
else:
    raise NotImplementedError

if use_wandb:
    config = {
        "policy_type": model_name,
        "total_timesteps": total_timesteps,
        "env_name": ENV_NAME,
        "commit": COMMIT,
    }
    run = wandb.init(
        project="drone",
        name=f"{model_name}_{noise_level}",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=False,  # auto-upload the videos of agents playing the game
        save_code=False,  # optional
    )
    model.learn(
        total_timesteps=total_timesteps,
        callback=WandbCallback(
            verbose=2),
    )
else:
    model.learn(total_timesteps=total_timesteps)

model.save(f"./policy/{model_name}_{ENV_NAME}_{noise_level}_{total_timesteps}")
del model
