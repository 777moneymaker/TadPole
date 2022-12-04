import dill
import pandas as pd


#def metadata_dump():
"""
Dump metadata to a dill file
"""

#read tsv
df = pd.read_table('Data/metadata-phrog.tsv', header=0)

#write dictionary
metadata_dict = dict(zip(df['phrog_id'], df['category']))

#write to dill
dill.dump(metadata_dict, open('Data/metadata_phrog.dill', 'wb'))



