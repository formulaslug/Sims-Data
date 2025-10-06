import pandas as pd

csv_file_1 = 'slipData.csv' 
csv_file_2 = 'latSlipData.csv' 

df1 = pd.read_csv(csv_file_1)
df2 = pd.read_csv(csv_file_2)

columns_to_keep = ['V', 'N', 'SA', 'SR', 'NFX', 'NFY', 'FZ', 'TSTC', 'AmbTmp']

df1_filtered = df1[columns_to_keep]
df2_filtered = df2[columns_to_keep]

combined_df = pd.concat([df1_filtered, df2_filtered], ignore_index=True)

combined_df.to_csv('combinedSet.csv', index=False)

print("Combined CSV file created successfully.")
