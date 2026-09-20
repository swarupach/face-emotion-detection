import pandas as pd
data = pd.read_csv("dataset/fer2013.csv")
print("Dataset shape:")
print(data.shape)

print("\nColumn names:")
print(data.columns)

print("\nFirst 5 rows:")
print(data.head())

print("\nEmotion counts:")
print(data["emotion"].value_counts().sort_index())

print("\nUsage counts:")
print(data["Usage"].value_counts())
