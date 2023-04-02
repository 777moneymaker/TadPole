from pathlib import Path
import re
import time
import logging
import json

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


class Word2VecPipeline:
    __slots__ = ("corpus_path", "output_prefix", "metadata", "vector_size", "window",
                 "min_count", "epochs", "workers", "lr_start", "lr_min", "sg", "hg",
                 "callbacks", "negative", "ns_exp", "show_debug", "n_top_phrogs", "visualise_model",
                 "encoded", "result", "model_name", "model_object")

    def __init__(self, corpus_path: str, output_prefix: str, metadata: str, vector_size: int = 100,
                 window: int = 5, min_count: int = 5, epochs: int = 5, workers: int = 3,
                 lr_start: float = 0.025, lr_min: float = 0.0001, sg: int = 0, hs: int = 0,
                 callbacks=[TrainLogger()], negative: int = 5, ns_exp: float = 0.75, show_debug: bool = False,
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
        self.sg = sg
        self.hs = hs
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
    
    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     del state['model_object']
    #     return state

    def _make_summary(self):
        summary = {k: v for k, v in self.__dict__.items() if k != 'model_object'}
        with open(f"evaluation/{self.model_name}_summary.json", 'w') as f:
            json.dump(summary, f)
    
    # def _dump_result(self):
    #     with open(f"evaluation/{self.model_name}_summary.json", 'w') as f:
    #         json.dump(self, f)
    
    def _generate_name(self) -> str:
        ns_exp_str = str(self.ns_exp).replace(".", "")
        lr_start_str = str(self.lr_start).replace(".", "")
        lr_min_str = str(self.lr_min).replace(".", "")
        self.model_name = f"{self.output_prefix}_ns{ns_exp_str}_lr{lr_start_str}_lrmin{lr_min_str}_d{self.vector_size}_w{self.window}_e{self.epochs}_hs{self.hs}_neg{self.negative}_mincount{self.min_count}"

    def _model_train(self):
        logging.basicConfig(level=logging.ERROR)
        if self.show_debug:
            logging.basicConfig(level=logging.DEBUG)

        with alive_bar(title = "Loading corpus",  dual_line = True, spinner = PHROG_SPINNER) as bar:
            sentences = utils.read_corpus(Path(self.corpus_path))
            bar()

        with alive_bar(title = "Creating model",  dual_line = True, spinner = PHROG_SPINNER) as bar:
            model = wv.Word2Vec(
                vector_size=self.vector_size,
                window=self.window,
                min_count=self.min_count,
                epochs=self.epochs, 
                workers=self.workers,
                alpha=self.lr_start,
                min_alpha=self.lr_min,
                sg=self.sg,
                hs=self.hs,
                ns_exponent=self.ns_exp,
                negative=self.negative)
            model.build_vocab(sentences, progress_per=1000)
            model.train(corpus_iterable=sentences, 
                total_examples=model.corpus_count, 
                epochs=model.epochs,
                compute_loss=True,
                callbacks=self.callbacks)
            print(model.__dict__)
            bar()
            
        self.model_object = model
    
    def _evaluate_model(self):
        funcs = utils.read_metadata(Path(self.metadata))
        prediction = evl.prediction(func_dict=funcs, model=self.model_object, model_name=self.model_name, top_known_phrogs=self.n_top_phrogs)
        return prediction
    
    def _umap_reduce(self, vectors_obj: wv.KeyedVectors, n_dims: int):
        with alive_bar(title = "UMAP Magic",  dual_line = True, spinner = PHROG_SPINNER) as bar:
            # custom_logger.logger.info("UMAP Magic")
            reducer = umap.UMAP(n_components=n_dims)
            # data_to_reduce = dataset['vector'].to_list()
            data_to_reduce = vectors_obj.vectors
            # reduce dimensionality
            embedding = reducer.fit_transform(data_to_reduce)
            bar()
            return embedding

    def _visualiser(self, vectors_obj: wv.KeyedVectors, reduced_embed: np.ndarray, visual_path: str, encoded: bool):
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
        self._generate_name()
        self._model_train()
        if self.save_model:
            model_path = f"train_test/{self.model_name}.model"
            self.model_object.save(model_path)
        self.result = self._evaluate_model()
        if self.visualise_model:
            self._visualise_model()
        self._make_summary()



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
    min_count: int) -> str:
    ns_exp_str = str(ns_exp).replace(".", "")
    lr_start_str = str(lr_start).replace(".", "")
    lr_min_str = str(lr_min).replace(".", "")
    return f"{prefix}_ns{ns_exp_str}_lr{lr_start_str}_lrmin{lr_min_str}_d{vector_size}_w{window}_e{epochs}_hs{hs}_neg{negative}_mincount{min_count}"


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
                    visual_path: str,
                    encoded: bool):

    """
    
    Generates model visualisation in plotly's 3D scatter.

    """

    with alive_bar(title = "Gathering phrog metadata and embedding data",  dual_line = True, spinner = PHROG_SPINNER) as bar:
        # func = utils.read_metadata(Path("Data/metadata_phrog.pickle"))
        dataset = pd.DataFrame({'word': vectors_obj.index_to_key})
        if not encoded:
            func = utils.read_metadata(Path("Data/metadata_phrog.pickle"))
        else:
            # func = utils.read_metadata(Path("Data/metadata_phrog_encoded.pickle"))
            func = utils.read_metadata(Path("Data/metadata_phrog_coded.pickle"))
        dataset["function"] = dataset['word'].map(func)
        dataset[['x', 'y', 'z']] = pd.DataFrame(reduced_embed, index=dataset.index)
        # dataset.to_string('plots/visual_df_diag_coded.txt')
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
    visualise_model: bool = False,
    encoded: bool = True):

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
        negative=negative,
        min_count=min_count)
    model_path = f"train_test/{model_name}.model"
    model.save(model_path)

    # *** lookup phrogs ***
    # lookup = utils.read_lookup_metadata(Path("Data/metadata_lookup_phrog.pickle"))
    # print(model.wv.index_to_key)
    # # for k,v in model.wv.index_to_key.items():
    # #     if v in lookup:
    # #         model.wv.index_to_key[k] = lookup[v]
    # for index, word in enumerate(model.wv.index_to_key):
    #     if word in lookup:
    #         model.wv.index_to_key[index] = lookup[word]
    
    # print(model.wv.index_to_key)
    # print(model.wv.most_similar("phrog_1512"))

    # *** model evaluation ***
    # this should be refactored before optimisation
    if not encoded:
        funcs = utils.read_metadata(Path("Data/metadata_phrog.pickle"))
        # funcs = utils.read_metadata(Path("Data/metadata_filtered_noncoded.pickle"))
    else:
        # funcs = utils.read_metadata(Path("Data/metadata_phrog_encoded.pickle"))
        funcs = utils.read_metadata(Path("Data/metadata_phrog_coded.pickle"))
    prediction = evl.prediction(func_dict=funcs, model=model, model_name=model_name)

    # *** visualise ***
    if visualise_model:
        visual_path = f"plots/{model_name}.html"
        embedding = umap_reduce(model.wv, n_dims=3)
        dataset = model_visualise(model.wv, embedding, visual_path, encoded)

