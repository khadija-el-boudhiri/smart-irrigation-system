import pandas as pd
from src.data_loader import load_dataset

def load_data(path):
    return pd.read_csv(path)