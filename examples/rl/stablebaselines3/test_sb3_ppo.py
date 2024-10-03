import gym
import gym_multirotor
from stable_baselines3 import PPO


# ENV_NAME = "QuadrotorXHoverEnv-v0"
# ENV_NAME = "TiltrotorPlus8DofHoverEnv-v0"
ENV_NAME = "QuadrotorPlusHoverEnv-v0"

model = PPO.load(f"./policy/PPO_{ENV_NAME}")
env = gym.make(ENV_NAME)
obs = env.reset()
render = False

while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    if render:
        env.render()
    if dones:
        obs = env.reset()