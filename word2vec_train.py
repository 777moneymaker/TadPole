from gensim.models import word2vec as wv
import numpy as np
import pandas as pd
import umap
import plotly.express as px
import dill
import pickle
import utils
import plotly
from pathlib import Path
import custom_logger
import time
import re


#TODO: make this whole file like a proper module

# start time
tic = time.perf_counter()

#  *** Word2vec ***
# corpus
#read pickle
# sentences = utils.read_corpus(Path("results/vir2000_collapsed.pickle"))
sentences = utils.read_corpus(Path("results/vir2000_numbered.pickle"))
# are jokers numbered?; btw this is not ideal, but it will all be changing
numbered = True

# train model (simplest form possible)
custom_logger.logger.info("Creating Word2vec model")
model = wv.Word2Vec(sentences, min_count=1, workers=4)

# save model to binary file
# model.save("train_test/test.model")
# print("Save model")

# get embedded vectors from the model
vectors = model.wv

# save embedded vectors to binary file
# vectors.save("train_test/test.wordvectors")
# print("Save vectors")

#  *** junk - skip ***
# d = np.vectorize(vectors.index_to_key.get)(vectors.vectors)
# print(d)
# dataset = pd.DataFrame(np.hstack((vectors.index_to_key, vectors.vectors.reshape(-1, 1))))
# print(dataset)
#  *******************

#  *** UMAP ***
# initialise UMAP
custom_logger.logger.info("UMAP Magic")
reducer = umap.UMAP(n_components=3)
# data_to_reduce = dataset['vector'].to_list()
data_to_reduce = vectors.vectors
# reduce dimensionality
embedding = reducer.fit_transform(data_to_reduce)
# list of embeddings with reduced dims to n=3
# print(embedding)

#  *** Visualisation ***
# get functions from metadata
custom_logger.logger.info("Loading phrog metadata")
func = utils.read_metadata(Path("Data/metadata_phrog.dill"))
# gather data to dataframe
custom_logger.logger.info("Gathering data for visualisation")
dataset = pd.DataFrame({'word': vectors.index_to_key})
#add joker to func
if not numbered:
    func['joker']  = 'joker_placeholder'
# silly joker name enhancing part
if numbered:
    pattern = re.compile(r"joker\d+")
    jokers = [x for l in [list(filter(pattern.match, elem)) for elem in sentences] for x in l]
    print(jokers)
    joker_funcs = {joker: "joker_placeholder" for joker in jokers}
    func.update(joker_funcs)
# print(func)
# map functions to words
dataset["function"] = dataset['word'].map(func)
# insert embedding data
dataset[['x', 'y', 'z']] = pd.DataFrame(embedding, index=dataset.index)
# print(dataset)

# show plot
custom_logger.logger.info("Creating visualisation")
fig = px.scatter_3d(dataset, x='x', y='y', z='z', color='function', hover_data=["word"])
plotly.offline.plot(fig, filename='plots/vir2000.html')
fig.show()

toc = time.perf_counter()
elapsed_time = toc - tic
custom_logger.logger.info(f"Done in {elapsed_time:0.8f}")

