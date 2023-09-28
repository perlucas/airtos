import gymnasium as gym
import gym_anytrading
from gym_anytrading.envs import StocksEnv
from utils import load_dataset
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C, PPO
from custom_callback import CustomCallback
from custom_logger import configure_custom_logger
from sb3_contrib import RecurrentPPO

# StocksEnv.metadata['render_fps'] = 2.5


df=load_dataset(name='./AMZN.csv')
gym_maker_fn = lambda: gym.make(
    'stocks-v0',
    df=df,
    window_size=10,
    frame_bound=(10, 2000),
    # render_mode="human"
)

env = DummyVecEnv([gym_maker_fn])
# env.render_mode="human"

# Training
# model = A2C("MlpPolicy", env, verbose=1)
model = RecurrentPPO("MlpLstmPolicy", env, verbose=1)

record_callback = CustomCallback(verbose=1, n_steps=3)
custom_logger = configure_custom_logger(verbose=1, 
    tensorboard_log=None, 
    tb_log_name="A2C", 
    reset_num_timesteps=True,
    callback=record_callback)
model.set_logger(custom_logger)

model.learn(total_timesteps=500000, callback=record_callback)
# model.learn(total_timesteps=1000000, callback=record_callback)


# Save model
# model.save('./models/A2C')
model.save('./models/PPO_v2')

# Testing
# env = DummyVecEnv([lambda: gym.make(
#     'stocks-v0',
#     df=df,
#     window_size=10,
#     frame_bound=(500, 650),
#     render_mode="human"
# )])

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, done, info = env.step(action)

#     env.render("human")
#     if done:
#         print("info:", info)
#         break

# plt.cla()
# env.unwrapped.render_all()
# plt.show()