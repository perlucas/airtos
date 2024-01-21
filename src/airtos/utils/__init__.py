import pandas as pd


def load_dataset(name, index_name='Date'):
    '''
    Load the contents of the given CSV stock prices file into a brand new pandas dataset
    '''
    df = pd.read_csv(name, parse_dates=True, index_col=index_name)
    return df
