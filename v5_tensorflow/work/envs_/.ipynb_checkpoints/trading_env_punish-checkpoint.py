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

import matplotlib.pyplot as plt


# Actions constant definitions
ACTION_NOOP = 0
ACTION_WEAK_BUY = 1
ACTION_REGULAR_BUY = 2
ACTION_STRONG_BUY = 3
ACTION_WEAK_SELL = 4
ACTION_REGULAR_SELL = 5
ACTION_STRONG_SELL = 6

# Percentages by action
ACTION_PERCENTAGES = [
    0,      # noop
    0.05,   # weak buy
    0.1,    # regular buy
    0.25,   # strong buy
    0.05,   # weak sell
    0.1,    # regular sell
    0.25,   # strong sell
]

# Colors for each action, mapped by index
COLOR_CODES = [
    None,       # noop
    '#6AF26F',  # weak buy
    '#59C05C',  # regular buy
    '#2D8930',  # strong buy
    '#F97373',  # weak sell
    '#F53D3D',  # regular sell
    '#C30000',  # strong sell
]


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
        # new: reshape into array
        num_inputs = window_size * self.signal_features.shape[1] + 1
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(num_inputs,),
            dtype=np.float32,
            minimum=[0] * num_inputs,
            maximum=[1000] * num_inputs,
            name='observation')

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=ACTION_NOOP, maximum=ACTION_STRONG_SELL, name='action')

        # Initialize episode
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        # Initial budget covers 100 shares
        self._initial_funds = 100 * self.prices[self._start_tick]
        self._final_funds = None
        self._current_tick = None
        self._shares = None
        self._budget = None
        self._episode_ended = None
        self._history = None

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def render(self, mode="human"):
        if mode != "human":
            raise NotImplementedError(
                'not human mode has not been implemented')

        def _plot_position(position, tick):
            color = None
            if position == Positions.Short:
                color = 'red'
            elif position == Positions.Long:
                color = 'green'
            if color:
                plt.scatter(tick, self.prices[tick], color=color)

        # clear plot; render prices as reference
        plt.figure(figsize=(15, 8))
        plt.cla()
        plt.plot(self.prices)

        # render each position from history
        _tick = self._start_tick
        for past_action in self._history:
            color = COLOR_CODES[past_action]
            if color is not None:
                plt.scatter(_tick, self.prices[_tick], color=color)
            _tick += 1

        # add info as subtitles
        plt.suptitle(
            "Initial Funds: %.2f" % self._initial_funds + ' ~ ' +
            "Final Funds: %.2f" % self._final_funds
        )
        plt.show()

    def _reset(self):
        self._history = []
        self._current_tick = self._start_tick
        self._shares = 0
        self._budget = self._initial_funds
        self._episode_ended = False

        observation = self._get_observation()
        return ts.restart(np.array(observation, dtype=np.float32))

    def _step(self, action):
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        self._current_tick += 1


        step_reward = self._update_and_get_reward(action)
        # print(f"action: {action}, reward: {step_reward}")

        if self._current_tick == self._end_tick or not self._can_still_operate():
            self._episode_ended = True

        observation = self._get_observation()

        if self._episode_ended:
            self._final_funds = self._budget + \
                self._shares * self.prices[self._current_tick]
            return ts.termination(np.array(observation, dtype=np.float32), reward=step_reward)

        return ts.transition(np.array(observation, dtype=np.float32), reward=step_reward, discount=1.0)

    def _get_observation(self):
        num_inputs = self.window_size * self.signal_features.shape[1]
        signals = np.reshape(self.signal_features[(self._current_tick-self.window_size+1):self._current_tick+1], (num_inputs, ))
        earnings = (self._budget + self._shares * self.prices[self._current_tick])/self._initial_funds
        return np.append(signals, [earnings])

    def _process_data(self):
        prices = self.df.loc[:, 'Close'].to_numpy()

        # validate index (TODO: Improve validation)
        prices[self.frame_bound[0] - self.window_size]

        # get the actual prices within observed frame
        # ensure there are at least window_size ticks before the first observed one
        prices = prices[self.frame_bound[0] - self.window_size : self.frame_bound[1]]

        # get prices to range (0, 1]
        normalized_prices = (prices-np.min(prices)) / \
            (np.max(prices)-np.min(prices))

        # volume set
        volume = self.df.loc[:, 'Volume'].to_numpy()
        volume = volume[self.frame_bound[0] - self.window_size : self.frame_bound[1]]
        volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))

        signal_features = np.column_stack((normalized_prices, volume, ))

        return prices.astype(np.float32), signal_features.astype(np.float32)

    def _update_and_get_reward(self, action):

        # compute funds before action was taken
        prev_price = self.prices[self._current_tick - 1]
        prev_funds = self._budget + self._shares * prev_price

        # apply action to update state: budget and shares
        new_shares, operation_shares = self._compute_new_shares(
            action, price=prev_price)
        new_budget = self._compute_new_budget(
            action, operation_shares=operation_shares, price=prev_price)

        self._history.append(action if new_shares !=
                             self._shares else ACTION_NOOP)

        assert new_budget >= 0, f"New budget is negative: {new_budget}"
        assert new_shares >= 0, f"New shares is negative: {new_shares}"

        self._shares = new_shares
        self._budget = new_budget
        cur_funds = self._budget + self._shares * \
            self.prices[self._current_tick]

        coef = 1 if cur_funds < prev_funds else 100

        return (cur_funds - prev_funds) * coef if action != ACTION_NOOP else -100

    def _compute_new_shares(self, action, price):
        if action == ACTION_NOOP:
            return (self._shares, 0)

        is_sell = action in [ACTION_WEAK_SELL,
                             ACTION_REGULAR_SELL, ACTION_STRONG_SELL]
        # cannot sell actions if has not bought any before
        if self._shares == 0 and is_sell:
            return (0, 0)

        budget = np.max([0, self._budget])  # in case it's negative
        operation_shares = np.floor(
            (budget * ACTION_PERCENTAGES[action]) / price)  # number of actions to operate with

        if is_sell:
            # cannot sell more than it has
            operation_shares = np.min([operation_shares, self._shares])
            return (self._shares - operation_shares, operation_shares)

        # it's buy
        return (self._shares + operation_shares, operation_shares)

    def _compute_new_budget(self, action, operation_shares, price):
        if action == ACTION_NOOP:
            return self._budget

        is_sell = action in [ACTION_WEAK_SELL,
                             ACTION_REGULAR_SELL, ACTION_STRONG_SELL]
        if is_sell:
            return self._budget + price * operation_shares
        return self._budget - price * operation_shares

    def _can_still_operate(self):
        budget = np.max([0, self._budget])  # in case it's negative
        min_price = np.min(self.prices[self._current_tick:])
        return np.floor(budget * ACTION_PERCENTAGES[ACTION_WEAK_BUY] / min_price) > 0
