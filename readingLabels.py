import pandas as pd

labels_df = pd.read_csv('labels.csv') #csv file

# Convert to a dictionary
labels = dict(zip(labels_df['ClassID'], labels_df['Name']))

print(labels)