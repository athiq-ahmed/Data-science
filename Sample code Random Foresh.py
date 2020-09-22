import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

RSEED = 50

# Load in data
df = pd.read_csv('https://s3.amazonaws.com/projects-rf/clean_data.csv')
df.head()

# Full dataset: https://www.kaggle.com/cdc/behavioral-risk-factor-surveillance-system