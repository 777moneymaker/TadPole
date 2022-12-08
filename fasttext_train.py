from pathlib import Path
import re
import time

from gensim.models import FastText
import gensim.models.fasttext
import numpy as np
import pandas as pd
import umap
import plotly
import plotly.express as px
from alive_progress import alive_bar
from alive_progress.animations.spinners import bouncing_spinner_factory


import utils
import custom_logger


PHROG_SPINNER = bouncing_spinner_factory(("🐸", "🐸"), 8, background = ".", hide = False, overlay =True)


@utils.time_this
def visualisation_pipeline(
    corpus_path: str,
    visual_path: str,
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 5,
    epochs: int = 5,
    workers: int = 3,
    lr_start: float = 0.025,
    lr_min: float = 0.0001,
    max_n: int = 3,
    min_n: int = 6,
    sg: int = 0, 
    hs: int = 0,
    sorted_vocab: int = 1
    ):
    """
    Automated fasttext pipeline: model training, UMAP dimensionality reduction, 3D scatter visualisation.
    """
    # *** fasttext train + loading corpus ***
    model = model_train(
        corpus_path, 
        vector_size, 
        window, 
        min_count, 
        epochs, 
        workers, 
        lr_start,
        lr_min,
        max_n,
        min_n,
        sg,
        hs,
        sorted_vocab)
    # print(type(model.wv))
    print(model.wv.vector_size)
    print(model.epochs)

    #  *** UMAP ***
    embedding = umap_reduce(model.wv, n_dims=3)
    # print(type(embedding))
    # print(embedding)

    #  *** Visualisation ***
    dataset = model_visualise(model.wv, embedding, visual_path)
    # print(dataset)


def model_visualise(vectors_obj: gensim.models.fasttext.FastTextKeyedVectors, reduced_embed: np.ndarray, visual_path: str):
    """
    Generates model visualisation in plotly's 3D scatter. 
    """
    with alive_bar(title = "Gathering phrog metadata and embedding data",  dual_line = True, spinner = PHROG_SPINNER) as bar:
        # custom_logger.logger.info("Loading phrog metadata")
        func = utils.read_metadata(Path("Data/metadata_phrog.pickle"))
        # gather data to dataframe
        # custom_logger.logger.info("Gathering data for visualisation")
        dataset = pd.DataFrame({'word': vectors_obj.index_to_key})
        
        #add joker to func
        # if not numbered:
        #     func['joker']  = 'joker_placeholder'
        # # silly joker name enhancing part
        # if numbered:
        #     pattern = re.compile(r"joker\d+")
        #     jokers = [x for l in [list(filter(pattern.match, elem)) for elem in sentences] for x in l]
        #     # print(jokers)
        #     joker_funcs = {joker: "joker_placeholder" for joker in jokers}
        #     func.update(joker_funcs)

        # map functions to words
        dataset["function"] = dataset['word'].map(func)

        # insert embedding data
        dataset[['x', 'y', 'z']] = pd.DataFrame(reduced_embed, index=dataset.index)
        bar()
    with alive_bar(title = "Generating visualisation",  dual_line = True, spinner = PHROG_SPINNER) as bar:
        fig = px.scatter_3d(dataset, x='x', y='y', z='z', color='function', hover_data=["word"])
        fig.write_html(Path(visual_path).as_posix())
        bar()


def umap_reduce(vectors_obj: gensim.models.fasttext.FastTextKeyedVectors, n_dims: int):
    """
    Uses UMAP to reduce the dimensionality of the embedded vectors.
    """
    with alive_bar(title = "UMAP Magic",  dual_line = True, spinner = PHROG_SPINNER) as bar:
        # custom_logger.logger.info("UMAP Magic")
        reducer = umap.UMAP(n_components=n_dims)
        # data_to_reduce = dataset['vector'].to_list()
        data_to_reduce = vectors_obj.vectors
        # reduce dimensionality
        embedding = reducer.fit_transform(data_to_reduce)
        bar()
    return embedding


def model_train(
    corpus_path: str, 
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 5,
    epochs: int = 5,
    workers: int = 3,
    lr_start: float = 0.025,
    lr_min: float = 0.0001,
    max_n: int = 3,
    min_n: int = 6,
    sg: int = 0, 
    hs: int = 0,
    sorted_vocab: int = 1
    ):
    """
    Train fasttext model.
    """
    # load corpus
    with alive_bar(title = "Loading corpus",  dual_line = True, spinner = PHROG_SPINNER) as bar:
        sentences = utils.read_corpus(Path(corpus_path))
        bar()

    # train model
    # custom_logger.logger.info("Creating fasttext model")
    with alive_bar(title = "Creating model",  dual_line = True, spinner = PHROG_SPINNER) as bar:
        model = FastText(
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            sentences=sentences,
            epochs=epochs,
            workers=workers,
            alpha=lr_start,
            min_alpha=lr_min,
            max_n=max_n,
            min_n=min_n,
            sg=sg,
            hs=hs,
            sorted_vocab=sorted_vocab)
        bar()
    return model
