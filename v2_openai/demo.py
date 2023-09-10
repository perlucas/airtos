import gym
from custom_env import register

register()
env = gym.make('CustomEnv-v0')

observation, _ = env.reset()
print(observation)
for _ in range(10):  # Assuming episode length is 10
    action = env.action_space.sample()  # Replace with your agent's action
    observation, reward, done, _, info = env.step(action)
    if done:
        break
