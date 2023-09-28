from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts


class Operation:
    def noop():
        return 0

    def buy():
        return 1

    def sell():
        return 2


class Strength:
    def weak():
        return 0

    def regular():
        return 1

    def strong():
        return 2


class TradingEnv(py_environment.PyEnvironment):

    def __init__(self, df, window_size, frame_bound):
        # Validate frame bound is a 2 dim tuple => (start, end)
        assert len(frame_bound) == 2
        self.frame_bound = frame_bound

        # Validate and process prices and signals from source data
        assert df.ndim == 2
        self.df = df
        self.window_size = window_size
        self.prices, self.signal_features = self._process_data()

        # Define observation and action space
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(window_size, self.signal_features.shape[1]), dtype=np.float32, minimum=0, maximum=1, name='observation')

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(2,), dtype=np.int32, minimum=0, maximum=2, name='action')

        # Initialize episode
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._current_tick = None
        self._shares = None
        self._budget = None
        self._episode_ended = None

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._current_tick = self._start_tick
        self._shares = 0
        # Initial budget covers 100 shares
        self._budget = 100 * self.prices[self._current_tick]
        self._episode_ended = False

        observation = self._get_observation()
        return ts.restart(np.array(observation, dtype=np.float32))

    def _step(self, action):
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._episode_ended = True

        step_reward = self._update_and_get_reward(action)

        observation = self._get_observation()

        if self._episode_ended:
            return ts.termination(np.array(observation, dtype=np.float32), reward=step_reward)

        return ts.transition(np.array(observation, dtype=np.int32), reward=step_reward, discount=1.0)

    def _get_observation(self):
        return self.signal_features[(self._current_tick-self.window_size+1):self._current_tick+1]

    def _process_data(self):
        prices = self.df.loc[:, 'Close'].to_numpy()

        # validate index (TODO: Improve validation)
        prices[self.frame_bound[0] - self.window_size]

        # get the actual prices within observed frame
        # ensure there are at least window_size ticks before the first observed one
        prices = prices[self.frame_bound[0] -
                        self.window_size:self.frame_bound[1]]

        # get prices to range (0, 1]
        normalized_prices = (prices-np.min(prices)) / \
            (np.max(prices)-np.min(prices))
        signal_features = np.column_stack((normalized_prices,))

        return prices.astype(np.float32), signal_features.astype(np.float32)

    def _update_and_get_reward(self, action):
        return 0.0
