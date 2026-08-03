# temporary script used to explore the structure of the dataset
import pandas as pd

df = pd.read_csv("data/01_raw/EurLex_all.csv", nrows=5)

print("\nColumns:\n")
print(df.columns.tolist())

print("\nFirst rows:\n")
print(df.head())