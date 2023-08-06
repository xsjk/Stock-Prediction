import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

data = pd.read_pickle('data.pkl')
print(data.dropna().shape)
print(data.shape)