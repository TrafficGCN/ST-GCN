import pandas as pd
import os

OS_PATH = os.path.dirname(os.path.realpath('__file__'))

# Load the pickle file
df = pd.read_pickle(OS_PATH + '/data/adj_mx_bay.pkl')

# Check if the loaded object is a list
if isinstance(df, list):
    # Convert the list to a DataFrame
    df = pd.DataFrame(df)

# Save to CSV
df.to_csv(OS_PATH + '/data/adj_mx_bay.csv', index=False)
