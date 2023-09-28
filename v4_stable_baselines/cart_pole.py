import gymnasium as gym

# from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

import numpy as np


def gym_maker_fn(): return gym.make(
    'CartPole-v1',
    render_mode='human'
)


env = DummyVecEnv([gym_maker_fn])

# env = make_vec_env('CartPole-v1', n_envs=2,
#    seed=1648070233)


# model = RecurrentPPO("MlpLstmPolicy", env=vec_env, verbose=1)
model = PPO("MlpPolicy", env=env, verbose=1,
            n_steps=32, n_epochs=20, batch_size=32, gae_lambda=0.8, gamma=0.98, learning_rate=0.001)
# model.learn(total_timesteps=1e5)
# model.save("qrdqn_cartpole")
model.load("qrdqn_cartpole")


obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render("human")
    if done:
        break


# TODO: test
# CartPole-v1:
#   n_envs: 8
#   n_timesteps: !!float 5e5
#   policy: 'MlpPolicy'
#   ent_coef: 0.0
