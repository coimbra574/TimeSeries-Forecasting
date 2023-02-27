import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose

os.chdir('/content/drive/MyDrive/Mestrado/Dados_ONS/Dados ONS')
df = pd.DataFrame()
for file in os.listdir():
    data = pd.read_excel(file)
    df = df.append(data)