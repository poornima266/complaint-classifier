import pandas as pd

# Load the file
df = pd.read_csv("complaints.csv")

# See the real column names first
print(df.columns.tolist())

