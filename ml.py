import pandas as pd
import numpy as np
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


df = pd.read_excel("Dataset/NIMS_FATIGUE.xlsx") 

# print(f"Shape: {df.shape} ") # Table format (x * x)
# print(f"\n--- Data types --- \n{df.dtypes}") # Data types
# nulls = df.isnull().sum()[lambda s: s > 0] # Empty cells
# print(f"\n--- Nulls --- \n{nulls if not nulls.empty else 'none'}")
# print(f"\n--- Duplicates --- \n{df.duplicated().sum()}")
# print(f"\n--- Numeric Summary ---\n{df.describe().T}") # count, mean, std, min, max, quartiles, Transpose
# print(f"\n--- Categorical / Low-cardinality ---") # if column is dtype object or less than 15 unique values, print name with count followed by 10 frequent values and count
# for c in df.columns:
#     n = df[c].nunique()
#     if df[c].dtype == "object" or n < 15:
#         print(f"\n{c}  ({n} unique)")
#         print(df[c].value_counts().head(10))


