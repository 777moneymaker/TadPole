from gensim.models import FastText
import numpy as np
import pandas as pd
import umap
import plotly.express as px
import re
import custom_logger
from pathlib import Path
import utils
import plotly
import time

#TODO: make this whole file like a proper module

# start time
tic = time.perf_counter()

#  *** fasttext ***
# corpus
#read pickle
# sentences = utils.read_corpus(Path("results/vir2000_collapsed.pickle"))
sentences = utils.read_corpus(Path("results/vir2000_numbered.pickle"))
# are jokers numbered?; btw this is not ideal, but it all will be changing 
numbered = True

#  *** junk - skip for now***
# silly joker name enhancing
# print(sentences)
# better_sentences = [[f"{inner_elem}{num1}{num2}" if inner_elem == 'joker' else inner_elem for num2, inner_elem in enumerate(elem)] for num1, elem in enumerate(sentences)]
# print("--------------------")
# print(better_sentences)
#  *******************

# instantiate fasttext (default parameters from gensim docs)
# model = FastText(vector_size=4, window=3, min_count=1)
# # build vocabulary
# model.build_vocab(corpus_iterable=sentences)

# train model
custom_logger.logger.info("Creating fasttext model")
model = FastText(vector_size=4, window=3, min_count=1, sentences=sentences, epochs=10)
# print(model.__dict__)
vectors = model.wv
# show everything about vectors
# print(vectors.__dict__)


#  *** UMAP ***
# initialise UMAP
custom_logger.logger.info("UMAP Magic")
reducer = umap.UMAP(n_components=3)
# data_to_reduce = dataset['vector'].to_list()
data_to_reduce = vectors.vectors
# reduce dimensionality
embedding = reducer.fit_transform(data_to_reduce)
# list of embeddings with reduced dims to n=3
print(embedding)


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

# map functions to words
dataset["function"] = dataset['word'].map(func)

# insert embedding data
dataset[['x', 'y', 'z']] = pd.DataFrame(embedding, index=dataset.index)
print(dataset)

# show plot
fig = px.scatter_3d(dataset, x='x', y='y', z='z', color='function', hover_data=["word"])
plotly.offline.plot(fig, filename='plots/vir2000.html')
fig.show()

toc = time.perf_counter()
elapsed_time = toc - tic
custom_logger.logger.info(f"Done in {elapsed_time:0.8f}")
