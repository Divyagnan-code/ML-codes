import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

data = pd.read_csv("adult_data.csv", delimiter=", ")
df = pd.DataFrame(data)
print(df.head())
