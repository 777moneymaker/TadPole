from gensim.models import FastText
import numpy as np
import pandas as pd
import umap
import plotly.express as px
import dill
import pickle


#  *** fasttext ***
# corpus

#read pickle
# TODO: pickle change to dill and maybe add to utils.py    
with open('results/result.pickle', 'rb') as f:
    sentences = pickle.load(f)

# # instantiate fasttext (default parameters from gensim docs)
# model = FastText(vector_size=4, window=3, min_count=1)
# # build vocabulary
# model.build_vocab(corpus_iterable=sentences)
# train model
model = FastText(vector_size=4, window=3, min_count=1, sentences=sentences, epochs=10)
# print(model.__dict__)
vectors = model.wv
# show everything about vectors
print(vectors.__dict__)


#  *** UMAP ***
# initialise UMAP
reducer = umap.UMAP(n_components=3)
# data_to_reduce = dataset['vector'].to_list()
data_to_reduce = vectors.vectors
# reduce dimensionality
embedding = reducer.fit_transform(data_to_reduce)
# list of embeddings with reduced dims to n=3
print(embedding)


#  *** Visualisation ***
# gather data to dataframe
dataset = pd.DataFrame({'word': vectors.index_to_key})
# map functions to words
#TODO: refactor to function
#read dill  with functions
with open('Data/metadata_phrog.dill', 'rb') as in_strm:
    func = dill.load(in_strm)

#add joker to func
func['joker']  = 'joker_placeholder'

dataset["function"] = dataset['word'].map(func)

# insert embedding data
dataset[['x', 'y', 'z']] = pd.DataFrame(embedding, index=dataset.index)
print(dataset)

# show plot
fig = px.scatter_3d(dataset, x='x', y='y', z='z', color='function', hover_data=["word"])
fig.show()