# import json

# import numpy as np
# import pandas as pd

# df = pd.read_csv("data/val.csv")
# # drop outcome
# df = df.drop(columns=["Outcome"])
# n = len(df)
# columns = df.columns.tolist()
# # create 100 requests
# request_list = []
# for i in range(100):
#     id = np.random.randint(1000, 100000)
#     num_data = np.random.randint(1, n // 2)
#     data = df.sample(num_data)
#     # convert to list
#     data = data.values.tolist()
#     request = {"id": id, "data": data, "columns": columns}
#     print(request)
#     save_path = f"requests/request_{id}.json"
#     with open(save_path, "w") as f:
#         json.dump(request, f, indent=4)

import json
import os
import numpy as np
import pandas as pd

# Create a folder requests if it doesn't exist
if not os.path.exists("requests"):
    os.mkdir("requests")

# Read the val.csv file
df = pd.read_csv("data/val.csv")

# Drop the outcome column
df = df.drop(columns=["Outcome"])

# Get the number of rows in the dataframe
n = len(df)

# Get the list of column names
columns = df.columns.tolist()

# Create 100 request files
request_list = []
for i in range(100):
    id = np.random.randint(1000, 100000)
    num_data = np.random.randint(1, n // 2)
    data = df.sample(num_data)

    # Convert the data to a list
    data = data.values.tolist()

    # Create the request
    request = {"id": id, "data": data, "columns": columns}

    # Save the request to a file
    save_path = f"requests/request_{id}.json"
    with open(save_path, "w") as f:
        json.dump(request, f, indent=4)

