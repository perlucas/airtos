import pandas_ta as ta
import numpy as np

from .trading_env import TradingEnv


class AdxEnv(TradingEnv):
    """Trading environment designed to use ADX as feature signals
    """

    def _process_data(self):
        prices = self.df.loc[:, 'Close'].to_numpy()

        # validate index (TODO: Improve validation)
        prices[self.frame_bound[0] - self.window_size]

        # Get the actual prices within observed frame
        # Ensure there are at least window_size ticks before the first observed one
        prices = prices[self.frame_bound[0] -
                        self.window_size: self.frame_bound[1]]

        # Generate indicators
        self.df.ta.log_return(cumulative=True, append=True)
        self.df.ta.percent_return(cumulative=True, append=True)

        adx = self.df.ta.adx().to_numpy()
        adx = np.where(np.isfinite(adx), adx, 0)

        return prices.astype(np.float32), adx.astype(np.float32)
