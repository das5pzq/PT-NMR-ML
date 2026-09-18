import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_parquet("Sample_vector.parquet")

signal = df.iloc[5, :500]

plt.plot(signal)
plt.show()