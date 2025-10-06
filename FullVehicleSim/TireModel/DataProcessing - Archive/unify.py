import pandas as pd

columns_to_append = ['ET', 'V', 'N', 'SA', 'SR', 'IA', 'RL', 'RE', 'P', 'FX', 'FY', 'FZ', 'MX', 'MZ', 'NFX', 'NFY', 'RST', 'TSTI', 'TSTC', 'TSTO']

df1 = pd.read_csv('saData.csv')
df2 = pd.read_csv('srData.csv')

df_combined = pd.concat([df1[columns_to_append], df2[columns_to_append]], ignore_index=True)

df_combined.to_csv('unifiedTraining.csv', index=False)