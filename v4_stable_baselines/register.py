from gymnasium.envs.registration import register

register(
    id='stocks-v1',
    entry_point='v4_stable_baselines.stocks_env:StocksEnv',
    kwargs={
        'df': None,
        'window_size': 10,
        'frame_bound': (10, 1000)
    }
)
