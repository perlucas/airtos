import pandas as pd

def load_dataset(name, index_name='Date'):
    # base_dir = os.path.dirname(os.path.abspath(__file__))
    # path = os.path.join(base_dir, 'data', name + '.csv')
    df = pd.read_csv(name, parse_dates=True, index_col=index_name)
    return df