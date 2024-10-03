import gym
import gym_multirotor
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import torch


SEED = 123
ENV_NAME = "QuadrotorPlusHoverEnv-v0"
num_envs = 1
total_timesteps = 5000000
model_name = "sac"

env = gym.make(ENV_NAME)  # ENV_NAMES = ["QuadrotorXHoverEnv-v0", "TiltrotorPlus8DofHoverEnv-v0", "QuadrotorPlusHoverEnv-v0"]

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
else:
    raise NotImplementedError

model.learn(total_timesteps=total_timesteps)
model.save(f"./policy/{model_name}_{ENV_NAME}_{total_timesteps}")
del model
