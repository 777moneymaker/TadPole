from glove import Corpus, Glove
import utils
from pathlib import Path
import pandas as pd
import plotly.express as px
import umap


sentences = utils.read_corpus(Path('results/vir2000_numbered.pickle'))
corpus = Corpus()
corpus.fit(sentences, window=5)
glove = Glove(no_components=50, learning_rate=0.01)
glove.fit(corpus.matrix, epochs=200, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)
# print(glove.word_vectors)
# print(glove.dictionary)
reducer = umap.UMAP(n_components=3)
# data_to_reduce = dataset['vector'].to_list()
data_to_reduce = glove.word_vectors
# reduce dimensionality
embedding = reducer.fit_transform(data_to_reduce)
func = utils.read_metadata(Path("Data/metadata_phrog.pickle"))
rev_dict = {v: k for k, v in glove.dictionary.items()}
dataset = pd.DataFrame({'word': rev_dict})
print(dataset)
dataset["function"] = dataset['word'].map(func)
dataset[['x', 'y', 'z']] = pd.DataFrame(embedding, index=dataset.index)
dataset = dataset.dropna()
fig = px.scatter_3d(dataset, x='x', y='y', z='z', color='function', hover_data=["word"], color_discrete_map=utils.colour_map)
fig.update_traces(marker_size = 4)
fig.write_html(Path('plots/vir2000_glove_test_v50_e200_lr001_w5.html').as_posix())
# glove.save('train_test/glove_test.model')
