from pathlib import Path
import re
import time
import logging
import os
import json
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


class FastTextPipeline(object):
    __slots__ = ("corpus_path", "output_prefix", "metadata", "vector_size", "window",
                 "min_count", "epochs", "workers", "lr_start", "lr_min", "max_n", "min_n",
                 "sg", "hs", "sorted_vocab", "callbacks", "negative", "ns_exp", "show_debug",
                 "n_top_phrogs", "visualise_model", "encoded", "result", "model_name", "model_object",
                 "save_model")

    def __init__(self, corpus_path: str, output_prefix: str, metadata: str, vector_size: int = 100,
                 window: int = 5, min_count: int = 5, epochs: int = 5, workers: int = 3,
                 lr_start: float = 0.025, lr_min: float = 0.0001, max_n: int = 3, min_n: int = 6, sg: int = 0, hs: int = 0,
                 sorted_vocab: int = 1, callbacks=[TrainLogger()], negative: int = 5, ns_exp: float = 0.75, show_debug: bool = False,
                 n_top_phrogs: int = 50, visualise_model: bool = False, encoded: bool = True, save_model: bool = True):
        self.corpus_path = corpus_path
        self.output_prefix = output_prefix
        self.metadata = metadata
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.workers = workers
        self.lr_start = lr_start
        self.lr_min = lr_min
        self.max_n = max_n
        self.min_n = min_n
        self.sg = sg
        self.hs = hs
        self.sorted_vocab = sorted_vocab
        self.callbacks = callbacks
        self.negative = negative
        self.ns_exp = ns_exp
        self.show_debug = show_debug
        self.n_top_phrogs = n_top_phrogs
        self.visualise_model = visualise_model
        self.encoded = encoded
        self.save_model = save_model
        self.result = None
        self.model_name = None
        self.model_object = None

    def _make_summary(self):
        # summary = {k: v for k, v in self.__dict__.items() if k != 'model_object'}
        summary = {attr: getattr(self, attr) for attr in self.__slots__ if attr not in ['model_object', 'callbacks']}
        Path("evaluation").mkdir(exist_ok=True)
        with open(f"evaluation/{self.model_name}_summary.json", 'w') as f:
            json.dump(summary, f)
    
    def _generate_name(self) -> str:
        ns_exp_str = str(self.ns_exp).replace(".", "")
        lr_start_str = str(self.lr_start).replace(".", "")
        lr_min_str = str(self.lr_min).replace(".", "")
        self.model_name = f"{self.output_prefix}_ns{ns_exp_str}_lr{lr_start_str}_lrmin{lr_min_str}_d{self.vector_size}_w{self.window}_e{self.epochs}_hs{self.hs}_neg{self.negative}_maxn{self.max_n}_minn{self.min_n}"
    
    def _model_train_exec(self):
        os.system(f"python fasttext_exec.py -c {self.corpus_path} -v {self.vector_size} -w {self.window} -m {self.min_count} -e {self.epochs} -t {self.workers} --lr {self.lr_start} --lr_min {self.lr_min} --max_n {self.max_n} --min_n {self.min_n} --sg {self.sg} --hs {self.hs} --sorted_vocab {self.sorted_vocab} --neg {self.negative} --ns_exp {self.ns_exp} --debug {self.show_debug} --model_name {self.model_name}")
        try:
            model_path = f"train_test/{self.model_name}.model"
            self.model_object = FastText.load(model_path)
            if not self.save_model:
                Path(model_path).unlink(missing_ok=True)
        except FileNotFoundError:
            print(2137)
            # return 2137
            # return
            raise ValueError("Model file not found")
    
    def _evaluate_model(self):
        funcs = utils.read_metadata(Path(self.metadata))
        print(self.model_object)
        prediction = evl.prediction(func_dict=funcs, model=self.model_object, model_name=self.model_name, top_known_phrogs=self.n_top_phrogs)
        return prediction
    
    def _umap_reduce(self, vectors_obj: gensim.models.fasttext.FastTextKeyedVectors, n_dims: int):
        with alive_bar(title = "UMAP Magic",  dual_line = True, spinner = PHROG_SPINNER) as bar:
            # custom_logger.logger.info("UMAP Magic")
            reducer = umap.UMAP(n_components=n_dims)
            # data_to_reduce = dataset['vector'].to_list()
            data_to_reduce = vectors_obj.vectors
            # reduce dimensionality
            embedding = reducer.fit_transform(data_to_reduce)
            bar()
            return embedding
    
    def _visualiser(self, vectors_obj: gensim.models.fasttext.FastTextKeyedVectors, reduced_embed: np.ndarray, visual_path: str, encoded: bool):
        with alive_bar(title = "Gathering phrog metadata and embedding data",  dual_line = True, spinner = PHROG_SPINNER) as bar:
            # func = utils.read_metadata(Path("Data/metadata_phrog.pickle"))
            dataset = pd.DataFrame({'word': vectors_obj.index_to_key})
            func = utils.read_metadata(Path(self.metadata))
            dataset["function"] = dataset['word'].map(func)
            dataset[['x', 'y', 'z']] = pd.DataFrame(reduced_embed, index=dataset.index)
            bar()
    
        with alive_bar(title = "Generating visualisation",  dual_line = True, spinner = PHROG_SPINNER) as bar:
            fig = px.scatter_3d(dataset, x='x', y='y', z='z', color='function', hover_data=["word"], color_discrete_map=utils.colour_map)
            fig.update_traces(marker_size = 4)
            fig.write_html(Path(visual_path).as_posix())
            bar()
    
    def _visualise_model(self):
        visual_path = f"plots/{self.model_name}.html"
        embedding = self._umap_reduce(self.model_object.wv, n_dims=3)
        dataset = self._visualiser(self.model_object.wv, embedding, visual_path, self.encoded)

    @utils.time_this
    def run(self):
        try:
            self._generate_name()
            self._model_train_exec()
            self.result = self._evaluate_model()
            if self.visualise_model:
                self._visualise_model()
            self._make_summary()
        except ValueError as e:
            self.result = None
            self.model_object = None
            raise


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
        funcs = utils.read_metadata(Path("Data/metadata_phrog.pickle"))
        # funcs = utils.read_metadata(Path("Data/metadata_filtered.pickle"))
    else:
        # funcs = utils.read_metadata(Path("Data/metadata_phrog_encoded.pickle"))
        funcs = utils.read_metadata(Path("Data/metadata_phrog_coded.pickle"))
        # funcs = utils.read_metadata(Path("Data/metadata_filtered.pickle"))
    prediction = evl.prediction(func_dict=funcs, model=model, model_name=model_name)

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
