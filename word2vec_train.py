from gensim.models import word2vec as wv
import numpy as np
import pandas as pd
import umap
import plotly.express as px


#  *** Word2vec ***
# corpus
# there will be binary object read with corpus as list of lists
sentences = [["cat", "say", "meow"], ["dog", "say", "woof"], ['cow', 'say', 'moo'], ['pen', 'pencil', 'ruler']]
# train model (simplest form possible)
model = wv.Word2Vec(sentences, min_count=1)
# save model to binary file
model.save("train_test/test.model")
print(model.wv.index_to_key)
# get embedded vectors from the model
vectors = model.wv
print(vectors.__dict__)
# get vector for word 'cat'
print(vectors['cat'])
# save embedded vectors to binary file
vectors.save("train_test/test.wordvectors")
print(vectors.most_similar("cat"))


#  *** junk - skip ***
# d = np.vectorize(vectors.index_to_key.get)(vectors.vectors)
# print(d)
# dataset = pd.DataFrame(np.hstack((vectors.index_to_key, vectors.vectors.reshape(-1, 1))))
# print(dataset)
#  *******************

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
# there must be a function to parse all functions of all phrogs and create map of it
d_funcs = {
    "say": "verb",
    "dog": "animal",
    "woof": "verb",
    "meow": "verb",
    "cat": "animal",
    "cow": "animal",
    "moo": "verb",
    "pen": "office",
    "pencil": 'office',
    'ruler': 'office'
}
dataset["function"] = dataset['word'].map(d_funcs)
# print(dataset)
# insert embedding data
dataset[['x', 'y', 'z']] = pd.DataFrame(embedding, index=dataset.index)
print(dataset)

# show plot
fig = px.scatter_3d(dataset, x='x', y='y', z='z', color='function', hover_data=["word"])
fig.show()
