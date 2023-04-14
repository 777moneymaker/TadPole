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
    """ Callback to print training stats after each epoch

    Attributes: CallbackAny2Vec

    Methods: on_epoch_end, on_train_end

    """

    def __init__(self):
        """ Initialise TrainLogger

        Args: None

        Returns: None

        """
        self.epoch = 0
        self.loss_to_show = 0
        self.success = 0

    # def on_train_begin(self, model):
    #     model.compute_loss = True
    
    def on_epoch_end(self, model):
        """ Print training stats after each epoch

        Args: model (gensim.models.word2vec.Word2Vec): Word2Vec model

        Returns: None

        """
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
        """ Print training stats after all epochs

        Args: model (gensim.models.word2vec.Word2Vec): Word2Vec model

        Returns: None

        """
        self.success = 1
        print("Actually finished all")


class Word2VecPipeline(object):
    """ Word2Vec pipeline
    
    Attributes: corpus_path (str): Path to corpus
                output_prefix (str): Prefix for output files
                metadata (str): Path to metadata file
                vector_size (int): Dimensionality of the feature vectors
                window (int): Maximum distance between the current and predicted word within a sentence
                min_count (int): Ignores all words with total frequency lower than this
                epochs (int): Number of iterations (epochs) over the corpus
                workers (int): Use these many worker threads to train the model (=faster training with multicore machines)
                lr_start (float): Initial learning rate
                lr_min (float): Minimal learning rate
                sg (int): Training algorithm: 1 for skip-gram; otherwise CBOW
                hs (int): If 1, hierarchical softmax will be used for model training. If set to 0, and negative is non-zero, negative sampling will be used.
                callbacks (list): List of callbacks to use during training

    Methods: __init__, _make_summary, _dump_result, _load_corpus, _load_metadata, _train_model, _visualise_model, _evaluate_model, _save_model, run

    """

    __slots__ = ("corpus_path", "output_prefix", "metadata", "vector_size", "window",
                 "min_count", "epochs", "workers", "lr_start", "lr_min", "sg", "hs",
                 "callbacks", "negative", "ns_exp", "show_debug", "n_top_phrogs", "visualise_model",
                 "encoded", "result", "model_name", "model_object", "save_model")

    def __init__(self, corpus_path: str, output_prefix: str, metadata: str, vector_size: int = 100,
                 window: int = 5, min_count: int = 5, epochs: int = 5, workers: int = 3,
                 lr_start: float = 0.025, lr_min: float = 0.0001, sg: int = 0, hs: int = 0,
                 callbacks=[TrainLogger()], negative: int = 5, ns_exp: float = 0.75, show_debug: bool = False,
                 n_top_phrogs: int = 50, visualise_model: bool = False, encoded: bool = True, save_model: bool = True):
        
        """ Initialise Word2VecPipeline
        
        Args: corpus_path (str): Path to corpus
                output_prefix (str): Prefix for output files
                metadata (str): Path to metadata file
                vector_size (int): Dimensionality of the feature vectors
                window (int): Maximum distance between the current and predicted word within a sentence
                min_count (int): Ignores all words with total frequency lower than this
                epochs (int): Number of iterations (epochs) over the corpus
                workers (int): Use these many worker threads to train the model (=faster training with multicore machines)
                lr_start (float): Initial learning rate
                lr_min (float): Minimal learning rate
                sg (int): Training algorithm: 1 for skip-gram; otherwise CBOW
                hs (int): If 1, hierarchical softmax will be used for model training. If set to 0, and negative is non-zero, negative sampling will be used.
                callbacks (list): List of callbacks to use during training

        Returns: None

        """

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
        """ Make summary of pipeline parameters

        Args: None

        Returns: None
        
        """

        # summary = {k: v for k, v in self.__dict__.items() if k != 'model_object'}
        summary = {attr: getattr(self, attr) for attr in self.__slots__ if attr not in ['model_object', 'callbacks']}
        Path("evaluation").mkdir(exist_ok=True)
        with open(f"evaluation/{self.model_name}_summary.json", 'w') as f:
            json.dump(summary, f)
    
    # def _dump_result(self):
    #     with open(f"evaluation/{self.model_name}_summary.json", 'w') as f:
    #         json.dump(self, f)
    
    def _generate_name(self) -> str:

        """ Generate name for model

        Args: None

        Returns: None

        """

        ns_exp_str = str(self.ns_exp).replace(".", "")
        lr_start_str = str(self.lr_start).replace(".", "")
        lr_min_str = str(self.lr_min).replace(".", "")
        self.model_name = f"{self.output_prefix}_ns{ns_exp_str}_lr{lr_start_str}_lrmin{lr_min_str}_d{self.vector_size}_w{self.window}_e{self.epochs}_hs{self.hs}_neg{self.negative}_mincount{self.min_count}"

    def _model_train(self):

        """ Train model
        
        Args: None

        Returns: None
        
        """

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

        """ Evaluate model
        
        Args: None

        Returns: None
        
        """

        funcs = utils.read_metadata(Path(self.metadata))
        prediction = evl.prediction(func_dict=funcs, model=self.model_object, model_name=self.model_name, top_known_phrogs=self.n_top_phrogs)
        return prediction
    
    def _umap_reduce(self, vectors_obj: wv.KeyedVectors, n_dims: int):
        """ Reduce dimensionality of vectors using UMAP

        Args:  vectors_obj (wv.KeyedVectors): Word2Vec model object
                n_dims (int): Number of dimensions to reduce to

        Returns: embedding (np.ndarray): Reduced dimensionality vectors

        Example: embedding = _umap_reduce(model, 2) 
        
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

    def _visualiser(self, vectors_obj: wv.KeyedVectors, reduced_embed: np.ndarray, visual_path: str, encoded: bool):

        """ Visualise model
        
        Args:  vectors_obj (wv.KeyedVectors): Word2Vec model object
                reduced_embed (np.ndarray): Reduced dimensionality vectors
                visual_path (str): Path to save visualisation
                encoded (bool): Whether the model was trained on encoded data

        Returns: None
        
        """

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
        """ Visualise model
        
        Args: None

        Returns: None
        
        """
        visual_path = f"plots/{self.model_name}.html"
        embedding = self._umap_reduce(self.model_object.wv, n_dims=3)
        dataset = self._visualiser(self.model_object.wv, embedding, visual_path, self.encoded)
    
    @utils.time_this
    def run(self):
        """ Run model
        
        Args: None

        Returns: None
        
        """
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

    """Generate model name
    
    Args:  prefix (str): Prefix for model name
            ns_exp (float): Negative sampling exponent
            lr_start (float): Starting learning rate
            lr_min (float): Minimum learning rate
            vector_size (int): Vector size
            window (int): Window size
            epochs (int): Number of epochs
            hs (int): Hierarchical softmax
            negative (int): Number of negative samples
            min_count (int): Minimum word count

    Returns: model_name (str): Model name

    """

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

    """ Train Word2Vec model

    Args:  corpus_path (str): Path to corpus
            min_count (int): Minimum word count
            workers (int): Number of workers
            vector_size (int): Vector size
            window (int): Window size
            sg (int): Skip-gram
            hs (int): Hierarchical softmax
            lr_start (float): Starting learning rate
            lr_min (float): Minimum learning rate
            epochs (int): Number of epochs
            negative (int): Number of negative samples
            ns_exp (float): Negative sampling exponent
            callbacks (list): List of callbacks
            show_debug (bool): Show debug messages

    Returns: model (wv.Word2Vec): Word2Vec model

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

    """ Reduce dimensionality of vectors using UMAP of ebeded vectors

    Args:  vectors_obj (wv.KeyedVectors): Word2Vec model
            n_dims (int): Number of dimensions to reduce to

    Returns: embedding (np.ndarray): Reduced dimensionality vectors

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

    """ Visualise model using UMAP reduced vectors in 3D by plotting with Plotly
    
    Args:  vectors_obj (wv.KeyedVectors): Word2Vec model
            reduced_embed (np.ndarray): Reduced dimensionality vectors
            visual_path (str): Path to save visualisation
            encoded (bool): Whether the model is encoded or not

    Returns: None

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

    """ Pipeline to train Word2Vec model and visualise using UMAP

    Args:  corpus_path (str): Path to corpus
            visual_path (str): Path to save visualisation
            vector_size (int): Dimensionality of word vectors
            window (int): Context window size
            min_count (int): Minimum frequency count of words
            epochs (int): Number of epochs to train for
            workers (int): Number of workers to use
            lr_start (float): Starting learning rate
            lr_min (float): Minimum learning rate
            sg (int): Skip-gram or CBOW
            hs (int): Hierarchical Softmax or Negative Sampling
            callbacks (list): List of callbacks to use
            negative (int): Number of negative samples
            ns_exp (float): Negative sampling exponent
            show_debug (bool): Whether to show debug messages

    Returns: None

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

    """ Pipeline to train Word2Vec model and evaluate using phrog dataset

    Args:  corpus_path (str): Path to corpus
            output_prefix (str): Prefix for output files
            vector_size (int): Dimensionality of word vectors
            window (int): Context window size
            min_count (int): Minimum frequency count of words
            epochs (int): Number of epochs to train for
            workers (int): Number of workers to use
            lr_start (float): Starting learning rate
            lr_min (float): Minimum learning rate
            sg (int): Skip-gram or CBOW
            hs (int): Hierarchical Softmax or Negative Sampling
            callbacks (list): List of callbacks to use
            negative (int): Number of negative samples
            ns_exp (float): Negative sampling exponent
            show_debug (bool): Whether to show debug messages
            n_top_phrogs (int): Number of top phrogs to evaluate
            visualise_model (bool): Whether to visualise model
            encoded (bool): Whether to use encoded phrog dataset

    Returns: None

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

