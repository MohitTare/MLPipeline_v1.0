import pandas as pd
import numpy as np

def csvConnector(path):
    df = pd.read_csv(path)
    return df