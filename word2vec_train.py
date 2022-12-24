from pathlib import Path
import re
import time
import logging

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
import evaluation as evl


PHROG_SPINNER = bouncing_spinner_factory(("🐸", "🐸"), 8, background = ".", hide = False, overlay =True)


class TrainLogger(CallbackAny2Vec):
    """
    Callback to print training stats after each epoch
    """

    def __init__(self):
        self.epoch = 0
        self.loss_to_show = 0
        self.success = 0

    # def on_train_begin(self, model):
    #     model.compute_loss = True
    
    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_current = loss - self.loss_to_show
        self.loss_to_show = loss
        # if self.epoch == 0:
        lr = model.min_alpha_yet_reached
        trained = model.train_count
        # print(f'lr after epoch {self.epoch}: {lr}')
        # print(f' after epoch {self.epoch}: {loss_current}')
        # else:
        #     print(f'Loss after epoch {self.epoch}: {loss - self.loss_previous_step}')
        print(f"epoch: {self.epoch} lr: {lr}\t loss: {loss_current}\t count: {trained}")
        # print(model._log_progress())
        self.epoch += 1
    
    def on_train_end(self, model):
        self.success = 1
        print("Actually finished all")


def _generate_name(
    prefix: str, 
    ns_exp: float, 
    lr_start: float, 
    lr_min: float, 
    vector_size: int, 
    window: int, 
    epochs: int, 
    hs: int, 
    negative: int) -> str:
    ns_exp_str = str(ns_exp).replace(".", "")
    lr_start_str = str(lr_start).replace(".", "")
    lr_min_str = str(lr_min).replace(".", "")
    return f"{prefix}_ns{ns_exp_str}_lr{lr_start_str}_lrmin{lr_min_str}_d{vector_size}_w{window}_e{epochs}_hs{hs}_neg{negative}"


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
    epochs: int = 5,
    negative: int = 5,
    ns_exp: float = 0.75,
    callbacks=[],
    show_debug: bool = False):

    """

    Train w2v model.

    """

    logging.basicConfig(level=logging.ERROR)
    if show_debug:
        logging.basicConfig(level=logging.DEBUG)
    
    with alive_bar(title = "Loading corpus",  dual_line = True, spinner = PHROG_SPINNER) as bar:
        sentences = utils.read_corpus(Path(corpus_path))
        bar()

    with alive_bar(title = "Creating model",  dual_line = True, spinner = PHROG_SPINNER) as bar:
        model = wv.Word2Vec(
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            # sentences=sentences,
            epochs=epochs, 
            workers=workers,
            alpha=lr_start,
            min_alpha=lr_min,
            sg=sg,
            hs=hs,
            ns_exponent=ns_exp,
            negative=negative)
        model.build_vocab(sentences,
             progress_per=1000)
        # print(model.corpus_count)
        # print(model.epochs)
        model.train(corpus_iterable=sentences, 
            total_examples=model.corpus_count, 
            epochs=model.epochs,
            # start_alpha=lr_start,
            # end_alpha=lr_min,
            compute_loss=True,
            callbacks=callbacks)
        print(model.__dict__)
        # print(callbacks[0].success)
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
        fig = px.scatter_3d(dataset, x='x', y='y', z='z', color='function', hover_data=["word"], color_discrete_map=utils.colour_map)
        fig.update_traces(marker_size = 4)
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
    hs: int = 0,
    callbacks=[],
    negative: int = 5,
    ns_exp: float = 0.75,
    show_debug: bool = False):

    """
    Automated word2vec visualisation pipeline: model training, UMAP dimensionality reduction, 3D scatter visualisation.
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
        hs=hs,
        negative=negative,
        ns_exp=ns_exp,
        callbacks=callbacks,
        show_debug=show_debug)
    # print(type(model.wv))
    print(model.wv.vector_size)
    print(model.epochs)
    # print(model.lifecycle_events)
    # train_success = model.callbacks[0].success
    # print(train_success)

    #  *** UMAP ***
    embedding = umap_reduce(model.wv, n_dims=3)
    # print(type(embedding))
    # print(embedding)

    #  *** Visualisation ***
    dataset = model_visualise(model.wv, embedding, visual_path)
    # print(dataset)


@utils.time_this
def evaluation_pipeline(
    corpus_path: str,
    output_prefix: str,
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 5,
    epochs: int = 5,
    workers: int = 3,
    lr_start: float = 0.025,
    lr_min: float = 0.0001,
    sg: int = 0, 
    hs: int = 0,
    callbacks=[],
    negative: int = 5,
    ns_exp: float = 0.75,
    show_debug: bool = False,
    n_top_phrogs: int = 1,
    visualise_model: bool = False):

    """
    Automated word2vec evaluation pipeline: model training, automated model save and model scoring.
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
        hs=hs,
        negative=negative,
        ns_exp=ns_exp,
        callbacks=callbacks,
        show_debug=show_debug)
    # print(type(model.wv))
    print(model.wv.vector_size)
    print(model.epochs)
    # print(model.lifecycle_events)

    # *** generate output files paths + save model ***
    model_name = _generate_name(
        prefix=output_prefix,
        ns_exp=ns_exp, 
        lr_start=lr_start, 
        lr_min=lr_min, 
        vector_size=vector_size, 
        window=window, 
        epochs=epochs, 
        hs=hs, 
        negative=negative)
    model_path = f"train_test/{model_name}.model"
    model.save(model_path)

    # *** lookup phrogs ***
    lookup = utils.read_lookup_metadata(Path("Data/metadata_lookup_phrog.pickle"))
    print(model.wv.index_to_key)
    # for k,v in model.wv.index_to_key.items():
    #     if v in lookup:
    #         model.wv.index_to_key[k] = lookup[v]
    for index, word in enumerate(model.wv.index_to_key):
        if word in lookup:
            model.wv.index_to_key[index] = lookup[word]
    
    print(model.wv.index_to_key)
    print(model.wv.most_similar("phrog_1512"))

    # # *** model evaluation ***
    # funcs = utils.read_metadata(Path("Data/metadata_phrog.pickle"))
    # prediction = evl.prediction(func_dict=funcs, model=model, top_known_phrogs=n_top_phrogs)

    # *** visualise ***
    if visualise_model:
        visual_path = f"plots/{model_name}.html"
        embedding = umap_reduce(model.wv, n_dims=3)
        dataset = model_visualise(model.wv, embedding, visual_path)

