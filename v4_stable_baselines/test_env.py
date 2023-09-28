import gymnasium as gym

import register
from utils import load_dataset
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from sb3_contrib import RecurrentPPO

# from custom_logger import configure_custom_logger
# from sb3_contrib import RecurrentPPO

df=load_dataset(name='./AMZN.csv')
gym_maker_fn = lambda: gym.make(
    'stocks-v1',
    df=df,
    window_size=10,
    # frame_bound=(2000, 2500),
    frame_bound=(1000, 1030),
    # render_mode="human"
)

env = DummyVecEnv([gym_maker_fn])
# env.render_mode="human"

# Testing
# model = RecurrentPPO("MlpLstmPolicy", env, verbose=1)
# model.load('./models/PPO')

obs = env.reset()
while True:
    # action, _states = model.predict(obs)
    action = np.reshape(env.action_space.sample(), (1, 2))
    obs, rewards, done, info = env.step(action)
    print(obs)
    if done:
        print("info:", info)
        break
    else:
        env.render("human")

import os
os.system("pause")