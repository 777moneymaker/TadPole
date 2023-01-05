from pathlib import Path
import re
import time
import logging
import os

from gensim.models import FastText
import gensim.models.fasttext
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

    def on_train_begin(self, model):
        model.compute_loss = True
        print(model.running_training_loss)
    
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
    negative: int,
    max_n: int,
    min_n: int) -> str:
    ns_exp_str = str(ns_exp).replace(".", "")
    lr_start_str = str(lr_start).replace(".", "")
    lr_min_str = str(lr_min).replace(".", "")
    return f"{prefix}_ns{ns_exp_str}_lr{lr_start_str}_lrmin{lr_min_str}_d{vector_size}_w{window}_e{epochs}_hs{hs}_neg{negative}_maxn{max_n}_minn{min_n}"


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
    sorted_vocab: int = 1,
    negative: int = 5,
    ns_exp: float = 0.75,
    callbacks=[],
    show_debug: bool = False):
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
        sorted_vocab,
        negative,
        ns_exp,
        callbacks,
        show_debug)
    # print(type(model.wv))
    # print(model.wv.vector_size)
    # print(model.epochs)
    # # print(model.lifecycle_events)
    # print(model.compute_loss)
    # print(model.min_alpha)
    # print(model.min_alpha_yet_reached)
    # print(model.alpha)
    # model.wo
    # print(model.__dict__)
    # model._log_progress
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
def evaluation_pipeline_exec(
    corpus_path: str,
    output_prefix: str,
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
    sorted_vocab: int = 1,
    negative: int = 5,
    ns_exp: float = 0.75,
    show_debug: bool = False,
    n_top_phrogs: int = 1,
    visualise_model: bool = False,
    encoded: bool = True):
    """
    Automated fasttext pipeline: model training, UMAP dimensionality reduction, 3D scatter visualisation.
    """


    model_name = _generate_name(
        prefix=output_prefix,
        ns_exp=ns_exp, 
        lr_start=lr_start, 
        lr_min=lr_min, 
        vector_size=vector_size, 
        window=window, 
        epochs=epochs, 
        hs=hs, 
        negative=negative,
        max_n=max_n,
        min_n=min_n)
    # *** fasttext train + loading corpus ***
    model_train_exec(
        corpus_path, 
        model_name,
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
        sorted_vocab,
        negative,
        ns_exp,
        show_debug)
    # print(type(model.wv))
    # print(model.wv.vector_size)
    # print(model.epochs)
    # # print(model.lifecycle_events)
    # print(model.compute_loss)
    # print(model.min_alpha)
    # print(model.min_alpha_yet_reached)
    # print(model.alpha)
    # model.wo
    # print(model.__dict__)
    # model._log_progress
    # train_success = model.callbacks[0].success
    # print(train_success)
    try:
        model_path = f"train_test/{model_name}.model"
        model = FastText.load(model_path)
    except FileNotFoundError:
        print(2137)
        return 2137

    # *** model evaluation ***
    # this should be refactored before optimisation
    if not encoded:
        # funcs = utils.read_metadata(Path("Data/metadata_phrog.pickle"))
        funcs = utils.read_metadata(Path("Data/metadata_filtered.pickle"))
    else:
        # funcs = utils.read_metadata(Path("Data/metadata_phrog_encoded.pickle"))
        # funcs = utils.read_metadata(Path("Data/metadata_phrog_coded.pickle"))
        funcs = utils.read_metadata(Path("Data/metadata_filtered.pickle"))
    prediction = evl.prediction(func_dict=funcs, model=model, top_known_phrogs=n_top_phrogs)

    #  *** Visualisation ***
    if visualise_model:
        visual_path = f"plots/{model_name}.html"
        embedding = umap_reduce(model.wv, n_dims=3)
        dataset = model_visualise(model.wv, embedding, visual_path, encoded)
    # print(dataset)


def model_visualise(vectors_obj: gensim.models.fasttext.FastTextKeyedVectors, reduced_embed: np.ndarray, visual_path: str, encoded: bool):
    """
    Generates model visualisation in plotly's 3D scatter. 
    """
    with alive_bar(title = "Gathering phrog metadata and embedding data",  dual_line = True, spinner = PHROG_SPINNER) as bar:
        # custom_logger.logger.info("Loading phrog metadata")
        # func = utils.read_metadata(Path("Data/metadata_phrog.pickle"))
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

        if not encoded:
            func = utils.read_metadata(Path("Data/metadata_phrog.pickle"))
        else:
            # func = utils.read_metadata(Path("Data/metadata_phrog_encoded.pickle"))
            func = utils.read_metadata(Path("Data/metadata_phrog_coded.pickle"))

        # map functions to words
        dataset["function"] = dataset['word'].map(func)

        # insert embedding data
        dataset[['x', 'y', 'z']] = pd.DataFrame(reduced_embed, index=dataset.index)
        bar()
    with alive_bar(title = "Generating visualisation",  dual_line = True, spinner = PHROG_SPINNER) as bar:
        fig = px.scatter_3d(dataset, x='x', y='y', z='z', color='function', hover_data=["word"], color_discrete_map=utils.colour_map)
        fig.update_traces(marker_size = 4)
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


def model_train_exec(
    corpus_path: str,
    model_name: str,
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
    sorted_vocab: int = 1,
    negative: int = 5,
    ns_exp: float = 0.75,
    show_debug: bool = False
    ):
    os.system(f"python fasttext_exec.py -c {corpus_path} -v {vector_size} -w {window} -m {min_count} -e {epochs} -t {workers} --lr {lr_start} --lr_min {lr_min} --max_n {max_n} --min_n {min_n} --sg {sg} --hs {hs} --sorted_vocab {sorted_vocab} --neg {negative} --ns_exp {ns_exp} --debug {show_debug} --model_name {model_name}")




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
    sorted_vocab: int = 1,
    negative: int = 5,
    ns_exp: float = 0.75,
    callbacks=[],
    show_debug: bool = False
    ):
    """
    Train fasttext model.
    """

    # logging.basicConfig(level=logging.ERROR)
    # if show_debug:
    #     logging.basicConfig(level=logging.DEBUG)
    ## dont know why it exits on high learning rate for negative sampling and why there is completely no answer 
    logging.basicConfig(level=logging.DEBUG)

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
            # sentences=sentences,
            epochs=epochs,
            workers=workers,
            alpha=lr_start,
            min_alpha=lr_min,
            max_n=max_n,
            min_n=min_n,
            sg=sg,
            hs=hs,
            negative=negative,
            ns_exponent=ns_exp,
            sorted_vocab=sorted_vocab)
        # model = FastText()
        print(model.__dict__)
        try:
            model.build_vocab(sentences)
        except:
            print("I shat myself")
        # print(model.corpus_count)
        # print(model.epochs)
        print(model.__dict__)
        try:
            model.train(corpus_iterable=sentences, 
                total_examples=model.corpus_count,
                total_words=model.corpus_total_words, 
                epochs=model.epochs,
                # start_alpha=lr_start,
                # end_alpha=lr_min,
                compute_loss=True,
                # report_delay=0.5,
                callbacks=callbacks)
        except TypeError: 
            print("I shat myself")
        model.lifecycle_events
        model.save("train_test/ft_test.model")
        bar()
    return model
