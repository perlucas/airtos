import gym
from gym import spaces
import numpy as np
from custom_bot import CustomBot

EPISODE_LENGTH = 10
ACTION_SPACE_LENGTH = 2
INITIAL_STOCK_PRICE = 1000
INITIAL_CASH = 10000


class CustomEnv(gym.Env):
    def __init__(self, observation_file):
        # Array of 10 price variations
        self.observation_space = spaces.Box(
            low=-5, high=5, shape=(EPISODE_LENGTH,), dtype=np.float32)
        # Array of 2 outputs: operation and num_stocks
        self.action_space = spaces.Box(
            low=0, high=1, shape=(ACTION_SPACE_LENGTH,), dtype=np.float32)
        self.current_step = 0
        self.current_observation_set = 0
        self.episode_length = EPISODE_LENGTH
        self.observation_file = observation_file
        self.observations = self.load_observations()
        self.bot = CustomBot(initial_stock_price=INITIAL_STOCK_PRICE,
                             initial_cash=INITIAL_CASH, initial_stocks=0)
        print(f"observations size: {len(self.observations)}")

    def load_observations(self):
        with open(self.observation_file, 'r') as f:
            lines = f.readlines()
            observation_episodes = []
            for line in lines:
                # Each line contains 100 price variations
                # Parse each line into an array of 100 values
                # Generate chunks of size 10 for each one
                observations = np.array_split([float(value)
                                               for value in line.strip().split(' ')], EPISODE_LENGTH)
                observation_episodes.append(observations)
        return observation_episodes

    def reset(self, *, seed=None, options=None):
        # Randomly pick an observations set (a chunks array)
        self.current_observation_set = np.random.randint(
            low=0, high=len(self.observations))
        # print(
        #     f"ENV has been reset, current obs set: {self.current_observation_set}")
        self.current_step = 0
        initial_observation = self.observations[self.current_observation_set][self.current_step]
        self.bot = CustomBot(initial_stock_price=INITIAL_STOCK_PRICE,
                             initial_cash=INITIAL_CASH, initial_stocks=0)
        return np.array(initial_observation, dtype=np.float32), {}

    def step(self, action):
        # This also ensures we don't get an array overflow error
        if self.current_step + 1 >= self.episode_length:
            done = True
        else:
            done = False

        # print(
        #     f"moved step, current is {self.current_step}, obs set: {self.current_observation_set}, action is: {action}")

        if done:
            next_observation = [0 for _ in range(10)]
        else:
            next_observation = self.observations[self.current_observation_set][self.current_step + 1]

        self.bot.update_stock_price(
            variations=self.observations[self.current_observation_set][self.current_step])

        reward = self.bot.get_funds() - INITIAL_CASH
        # print(f"reward: {reward}")

        self.bot.process_action(action)

        self.current_step += 1

        return np.array(next_observation, dtype=np.float32), reward, done, {}


def register():
    from gym.envs.registration import register
    register(
        id='CustomEnv-v0',
        entry_point='openai.custom_env:CustomEnv',
        kwargs={'observation_file': './training_patterns.txt'}
    )
