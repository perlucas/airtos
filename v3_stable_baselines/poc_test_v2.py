import gymnasium as gym
import gym_anytrading
from gym_anytrading.envs import StocksEnv
from utils import load_dataset
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C
from custom_callback import CustomCallback
from custom_logger import configure_custom_logger
from sb3_contrib import RecurrentPPO

# StocksEnv.metadata['render_fps'] = 2.5


df=load_dataset(name='./AMZN.csv')
gym_maker_fn = lambda: gym.make(
    'stocks-v0',
    df=df,
    window_size=10,
    # frame_bound=(2000, 2500),
    frame_bound=(1000, 1100),
    # render_mode="human"
)

env = DummyVecEnv([gym_maker_fn])
env.render_mode="human"

# Testing
model = RecurrentPPO("MlpLstmPolicy", env, verbose=1)
model.load('./models/PPO_v2')

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)

    if done:
        print("info:", info)
        break
    else:
        env.render("human")

import os
os.system("pause")


# plt.cla()
# env.unwrapped.render_all()
# plt.show()



# A2C --> info: 'total_reward': -1.2855110168457031, 'total_profit': 0.6863108754337506
# PPO --> info: 'total_reward': -3.66400146484375, 'total_profit': 0.6361249743733768