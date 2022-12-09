from pathlib import Path
import re
import time

from gensim.models import word2vec as wv
from gensim.models.callbacks import CallbackAny2Vec
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


def model_train(
    corpus_path: str, 
    min_count: int = 5, 
    workers: int = 8,
    vector_size: int = 25,
    window: int = 3,
    sg: int = 1,
    hs: int = 1,
    lr_start: float = 0.4,
    lr_min: float = 0.005,
    epochs: int = 5):

    """

    Train w2v model.

    """
    
    with alive_bar(title = "Loading corpus",  dual_line = True, spinner = PHROG_SPINNER) as bar:
        sentences = utils.read_corpus(Path(corpus_path))
        bar()

    with alive_bar(title = "Creating model",  dual_line = True, spinner = PHROG_SPINNER) as bar:
        model = wv.Word2Vec(
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            sentences=sentences,
            epochs=epochs, 
            workers=workers,
            alpha=lr_start,
            min_alpha=lr_min,
            sg=sg,
            hs=hs)
        bar()
        
    return model


def umap_reduce(
    vectors_obj: wv.KeyedVectors, 
    n_dims: int):

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


def model_visualise(vectors_obj: wv.KeyedVectors, 
                    reduced_embed: np.ndarray, 
                    visual_path: str):

    """
    
    Generates model visualisation in plotly's 3D scatter.

    """

    with alive_bar(title = "Gathering phrog metadata and embedding data",  dual_line = True, spinner = PHROG_SPINNER) as bar:
        func = utils.read_metadata(Path("Data/metadata_phrog.pickle"))
        dataset = pd.DataFrame({'word': vectors_obj.index_to_key})
        dataset["function"] = dataset['word'].map(func)
        dataset[['x', 'y', 'z']] = pd.DataFrame(reduced_embed, index=dataset.index)
        bar()
    
    with alive_bar(title = "Generating visualisation",  dual_line = True, spinner = PHROG_SPINNER) as bar:
        fig = px.scatter_3d(dataset, x='x', y='y', z='z', color='function', hover_data=["word"])
        fig.write_html(Path(visual_path).as_posix())
        bar()


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
    sg: int = 0, 
    hs: int = 0):

    """
    Automated fasttext pipeline: model training, UMAP dimensionality reduction, 3D scatter visualisation.
    """
    # *** w2v train + loading corpus ***
    model = model_train(
        corpus_path=corpus_path, 
        vector_size=vector_size, 
        window=window, 
        min_count=min_count, 
        epochs=epochs, 
        workers=workers, 
        lr_start=lr_start,
        lr_min=lr_min,
        sg=sg,
        hs=hs)
    # print(type(model.wv))
    print(model.wv.vector_size)
    print(model.epochs)
    # print(model.lifecycle_events)

    #  *** UMAP ***
    embedding = umap_reduce(model.wv, n_dims=3)
    # print(type(embedding))
    # print(embedding)

    #  *** Visualisation ***
    dataset = model_visualise(model.wv, embedding, visual_path)
    # print(dataset)