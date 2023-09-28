import gymnasium as gym
import gym_anytrading
from gym_anytrading.envs import StocksEnv
from utils import load_dataset
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C, PPO
from v3_stable_baselines.custom_callback import CustomCallback
from v3_stable_baselines.custom_logger import configure_custom_logger
from sb3_contrib import RecurrentPPO
import register


df=load_dataset(name='./AMZN.csv')
gym_maker_fn = lambda: gym.make(
    'stocks-v1',
    df=df,
    window_size=10,
    frame_bound=(10, 2000),
)

env = DummyVecEnv([gym_maker_fn])

# Training
model = RecurrentPPO("MlpLstmPolicy", env, verbose=1)

record_callback = CustomCallback(verbose=1, n_steps=3)
custom_logger = configure_custom_logger(verbose=1, 
    tensorboard_log=None, 
    tb_log_name="PPO", 
    reset_num_timesteps=True,
    callback=record_callback)
model.set_logger(custom_logger)

model.learn(total_timesteps=1000000, callback=record_callback)


# Save model
model.save('./models/PPO')