import pandas as pd

# Read the CSV file
df = pd.read_csv('Data/lateralDataSet.csv')

# Filter out rows where NFY is less than 0.05
df_filtered = df[df['NFY'] >= 0.05]

# Save the filtered DataFrame to a new CSV file
df_filtered.to_csv('Data/filteredLateralDataset.csv', index=False)