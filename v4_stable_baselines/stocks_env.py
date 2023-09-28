import numpy as np

from .trading_env import TradingEnv, Actions, Positions, StrengthIndicator


class StocksEnv(TradingEnv):

    def __init__(self, df, window_size, frame_bound, render_mode=None):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        super().__init__(df, window_size, render_mode)

        self.trade_fee_bid_percent = 0.01  # unit
        self.trade_fee_ask_percent = 0.005  # unit

    def _process_data(self):
        prices = self.df.loc[:, 'Close'].to_numpy()

        # validate index (TODO: Improve validation)
        prices[self.frame_bound[0] - self.window_size]
        prices = prices[self.frame_bound[0] -
                        self.window_size:self.frame_bound[1]]

        normalized_prices = np.insert(
            (prices-np.min(prices))/(np.max(prices)-np.min(prices)), 0, 0)
        signal_features = np.column_stack((normalized_prices,))

        print(prices)
        print(signal_features)

        return prices.astype(np.float32), signal_features.astype(np.float32)

    # def _calculate_reward(self, action):
    #     previous_price = self.signal_features[self._current_tick - 1][0]
    #     current_price = self.signal_features[self._current_tick][0]

    #     price_diff = current_price - previous_price
    #     # print([previous_price, current_price, price_diff, self._current_holding])

    #     return price_diff * self._current_holding

    def _new_holdings(self, action):
        decision, strength = action

        # holdings don't change if NOOP
        if decision == Actions.Noop.value:
            return self._current_holding

        if decision == Actions.Sell.value:
            to_sell = np.min(
                [self._current_holding, self._compute_tsx_holdings(strength)])
            return np.max([0, self._current_holding - to_sell])

        return self._current_holding + self._compute_tsx_holdings(strength)

    def _compute_tsx_holdings(self, strength):
        perc = [StrengthIndicator.Weak, StrengthIndicator.Regular,
                StrengthIndicator.Strong][strength].percentage()
        price = self.signal_features[self._current_tick - 1][0]
        if price == 0:
            return 0
        return np.max([0, np.around((self._current_budget * perc)/price) - 1])

    def _new_budget(self, action):
        # start from current_budget
        # add tsx_holdings that were sold
        # substract tsx_holdings that were bought

        print(f"current price: ${self.prices[self._current_tick]}")

        decision, strength = action
        previous_price = self.signal_features[self._current_tick - 1][0]

        if decision == Actions.Noop.value:
            return self._current_budget

        if decision == Actions.Sell.value:
            to_sell = np.min(
                [self._current_holding, self._compute_tsx_holdings(strength)])
            return self._current_budget + to_sell * previous_price

        to_buy = self._compute_tsx_holdings(strength)
        return np.max([0, self._current_budget - to_buy * previous_price])

    # def _update_profit(self, action):
    #     new_budget = self._new_budget(action)
    #     new_holding = self._new_holdings(action)
    #     self._current_budget = new_budget
    #     self._current_holding = new_holding

    def _update_and_get_reward(self, action):
        previous_price = self.signal_features[self._current_tick - 1][0]
        previous_assets = self._current_budget + self._current_holding * previous_price

        # update budgen and shares
        new_budget = self._new_budget(action)
        new_holding = self._new_holdings(action)
        self._current_budget = new_budget
        self._current_holding = new_holding
        current_price = self.signal_features[self._current_tick][0]
        current_assets = self._current_budget + self._current_holding * current_price

        return current_assets - previous_assets
