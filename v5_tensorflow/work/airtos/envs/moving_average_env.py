import pandas_ta as ta
import numpy as np

from .trading_env import TradingEnv


class MovingAverageEnv(TradingEnv):
    """Trading environment designed to use MAs as feature signals
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

        ma1 = self.df.ta.ema(length=20).to_numpy()
        ma1 = np.where(np.isfinite(ma1), ma1, 0)

        ma2 = self.df.ta.ema(length=200).to_numpy()
        ma2 = np.where(np.isfinite(ma2), ma2, 0)

        features = np.column_stack((ma1, ma2,))

        return prices.astype(np.float32), features.astype(np.float32)
