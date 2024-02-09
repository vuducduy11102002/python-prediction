import os
import numpy as np
import pandas as pd

# Create a folder csvPrediction if it doesn't exist
if not os.path.exists("csvPrediction"):
    os.mkdir("csvPrediction")

# Read the diabetes.csv file
df = pd.read_csv("data/diabetes.csv")

# Drop the outcome column
df = df.drop(columns=["Outcome"])

# Get the number of rows in the dataframe
n = len(df)

# Get the list of column names
columns = df.columns.tolist()

# Create 10 csvPrediction files
for i in range(10):
    id = np.random.randint(1000, 100000)
    num_data = np.random.randint(1, n // 2)
    data = df.sample(num_data)

    # Create the file name
    save_path = f"csvPrediction/csvPrediction_{id}.csv"

    # Save the data to a CSV file
    data.to_csv(save_path, index=False)
