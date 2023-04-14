from pathlib import Path
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
        """ Initialize the TrainLogger class

        Args: None

        Returns: None

        """
        self.epoch = 0
        self.loss_to_show = 0
        self.success = 0
    
    def on_epoch_end(self, model):
        """ Print training statistics after each epoch

        Args: model (gensim.models.word2vec.Word2Vec): Word2Vec model used for training.
        
        Returns: None

        """
        loss = model.get_latest_training_loss()
        loss_current = loss - self.loss_to_show
        self.loss_to_show = loss
        lr = model.min_alpha_yet_reached
        trained = model.train_count
        print(f"epoch: {self.epoch} lr: {lr}\t loss: {loss_current}\t count: {trained}")
        self.epoch += 1
    
    def on_train_end(self, model):
        """ Print final training statistics after all epochs

        This function is called after the training is completed for all epochs, and it prints
        the final training statistics. It sets the success flag to 1 and prints a message.

        Args: model (gensim.models.word2vec.Word2Vec): Word2Vec model used for training.

        Returns: None

        """
        self.success = 1
        print("Actually finished all")


class Word2VecPipeline(object):
    """ Word2Vec pipeline.

    This class implements a pipeline for training and evaluating Word2Vec models on a given corpus.

    Attributes:
        corpus_path (str): Path to the corpus file.
        output_prefix (str): Prefix for output files.
        metadata (str): Path to the metadata file.
        vector_size (int): Dimensionality of the feature vectors.
        window (int): Maximum distance between the current and predicted word within a sentence.
        min_count (int): Ignores all words with total frequency lower than this.
        epochs (int): Number of iterations (epochs) over the corpus.
        workers (int): Number of worker threads to use for training the model (for faster training with multicore machines).
        lr_start (float): Initial learning rate.
        lr_min (float): Minimal learning rate.
        sg (int): Training algorithm: 1 for skip-gram, 0 for CBOW.
        hs (int): If 1, hierarchical softmax will be used for model training. If set to 0, and negative is non-zero, negative sampling will be used.
        callbacks (list): List of callbacks to use during training.

    Methods:
        __init__: Initialize the Word2VecPipeline object.
        _make_summary: Generate a summary of the trained model.
        _dump_result: Dump the summary and results to output files.
        _load_corpus: Load the corpus from the specified path.
        _load_metadata: Load the metadata from the specified path.
        _train_model: Train the Word2Vec model on the loaded corpus.
        _visualise_model: Visualize the trained Word2Vec model.
        _evaluate_model: Evaluate the trained Word2Vec model.
        _save_model: Save the trained Word2Vec model to a file.
        run: Run the Word2Vec pipeline.
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
        
        """ Initializes Word2VecPipeline with the given parameters.

        Args: corpus_path (str): Path to the corpus.
            output_prefix (str): Prefix for output files.
            metadata (str): Path to the metadata file.
            vector_size (int): Dimensionality of the feature vectors.
            window (int): Maximum distance between the current and predicted word within a sentence.
            min_count (int): Ignores all words with total frequency lower than this.
            epochs (int): Number of iterations (epochs) over the corpus.
            workers (int): Number of worker threads to use for training the model (faster training with multicore machines).
            lr_start (float): Initial learning rate.
            lr_min (float): Minimal learning rate.
            sg (int): Training algorithm: 1 for skip-gram, 0 for CBOW.
            hs (int): If 1, hierarchical softmax will be used for model training. If set to 0, and negative is non-zero, negative sampling will be used.
            callbacks (list): List of callbacks to use during training.
            negative (int): Number of negative samples to use for negative sampling.
            ns_exp (float): Exponent used to shape the negative sampling distribution.
            show_debug (bool): Whether to show debug information during training.
            n_top_phrogs (int): Number of top phrogs to show during training.
            visualise_model (bool): Whether to visualize the model during training.
            encoded (bool): Whether the corpus is already encoded or not.
            save_model (bool): Whether to save the trained model or not.

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
    

    def _make_summary(self):
        """ Generates a summary of pipeline parameters.

        This method generates a summary of the pipeline parameters and saves it as a JSON file in the "evaluation" directory. The summary includes various parameters such as model_name, corpus_path, output_prefix, metadata, vector_size, window, min_count, epochs, workers, lr_start, lr_min, sg, hs, negative, ns_exp, show_debug, n_top_phrogs, visualise_model, encoded, and save_model.

        Args: 
            None

        Returns: 
            None
            
        """

        summary = {attr: getattr(self, attr) for attr in self.__slots__ if attr not in ['model_object', 'callbacks']}
        Path("evaluation").mkdir(exist_ok=True)
        with open(f"evaluation/{self.model_name}_summary.json", 'w') as f:
            json.dump(summary, f)
    
    
    def _generate_name(self) -> str:

        """Generates a name for the model based on pipeline parameters.

        This method generates a unique name for the model based on the values of various pipeline parameters, such as ns_exp, lr_start, lr_min, vector_size, window, epochs, hs, negative, and min_count. The generated name is used as the model_name attribute of the pipeline.

        Args: 
            None

        Returns: 
            str: The generated name for the model.
            
        """

        ns_exp_str = str(self.ns_exp).replace(".", "")
        lr_start_str = str(self.lr_start).replace(".", "")
        lr_min_str = str(self.lr_min).replace(".", "")
        self.model_name = f"{self.output_prefix}_ns{ns_exp_str}_lr{lr_start_str}_lrmin{lr_min_str}_d{self.vector_size}_w{self.window}_e{self.epochs}_hs{self.hs}_neg{self.negative}_mincount{self.min_count}"

    def _model_train(self):

        """Trains the Word2Vec model with specified parameters.

        This method trains the Word2Vec model using the specified parameters, such as vector_size, window, min_count, epochs, workers, alpha, min_alpha, sg, hs, ns_exponent, and negative. It reads the corpus from the corpus_path, creates the model, builds the vocabulary, and trains the model with the given callbacks. The trained model object is stored in the model_object attribute of the pipeline.

        Args:
            None

        Returns:
            None

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

        """Evaluates the Word2Vec model by making predictions.

        This method evaluates the trained Word2Vec model by making predictions using the given metadata file and the trained model object. It reads the metadata from the specified file, and calls the `prediction` function from the `evl` module to make predictions. The predictions are based on the functional annotations in the metadata, the trained model, and other parameters such as model_name and n_top_phrogs.

        Args:
            None

        Returns:
            dict: A dictionary containing the prediction results.

        """

        funcs = utils.read_metadata(Path(self.metadata))
        prediction = evl.prediction(func_dict=funcs, model=self.model_object, model_name=self.model_name, top_known_phrogs=self.n_top_phrogs)
        return prediction
    
    def _umap_reduce(self, vectors_obj: wv.KeyedVectors, n_dims: int):
        """Reduces dimensionality of vectors using UMAP.

        This method takes a trained Word2Vec model object and reduces the dimensionality of its vectors using the UMAP algorithm. The reduced-dimensional vectors are returned as a numpy array.

        Args:
            vectors_obj (wv.KeyedVectors): A trained Word2Vec model object.
            n_dims (int): The number of dimensions to reduce to.

        Returns:
            np.ndarray: A numpy array containing the reduced-dimensional vectors.

        Example:
            embedding = _umap_reduce(model, 2)

        """

        with alive_bar(title = "UMAP Magic",  dual_line = True, spinner = PHROG_SPINNER) as bar:
            reducer = umap.UMAP(n_components=n_dims)
            data_to_reduce = vectors_obj.vectors
            # reduce dimensionality
            embedding = reducer.fit_transform(data_to_reduce)
            bar()
            return embedding

    def _visualiser(self, vectors_obj: wv.KeyedVectors, reduced_embed: np.ndarray, visual_path: str, encoded: bool):

        """Visualizes model.

        This method takes a trained Word2Vec model object, reduced-dimensional vectors obtained from UMAP, and other parameters, and generates a 3D scatter plot visualization of the model. The visualization is saved as an HTML file.

        Args:
            vectors_obj (wv.KeyedVectors): A trained Word2Vec model object.
            reduced_embed (np.ndarray): A numpy array containing the reduced-dimensional vectors.
            visual_path (str): The path to save the visualization.
            encoded (bool): A boolean indicating whether the model was trained on encoded data.

        Returns:
            None

        """

        with alive_bar(title = "Gathering phrog metadata and embedding data",  dual_line = True, spinner = PHROG_SPINNER) as bar:
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
        """ Visualizes the Word2Vec model.

        This method generates a 3D scatter plot visualization of the Word2Vec model using UMAP to reduce the dimensionality of the word vectors. The visualization is saved as an HTML file.

        Args:
            None

        Returns:
            None

        """
        visual_path = f"plots/{self.model_name}.html"
        embedding = self._umap_reduce(self.model_object.wv, n_dims=3)
        dataset = self._visualiser(self.model_object.wv, embedding, visual_path, self.encoded)
    
    @utils.time_this
    def run(self):
        """ Runs the Word2Vec model.

        This method executes the training, evaluation, and visualization (optional) of the Word2Vec model. The trained model is saved (optional) and the summary of the model performance is generated.

        Args:
            None

        Returns:
            None

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

    """ Generates a name for the Word2Vec model.

    This method generates a name for the Word2Vec model based on the provided parameters.

    Args:
        prefix (str): Prefix for the model name.
        ns_exp (float): Negative sampling exponent.
        lr_start (float): Starting learning rate.
        lr_min (float): Minimum learning rate.
        vector_size (int): Vector size.
        window (int): Window size.
        epochs (int): Number of epochs.
        hs (int): Hierarchical softmax.
        negative (int): Number of negative samples.
        min_count (int): Minimum word count.

    Returns:
        model_name (str): Model name.

    """

    return f"{prefix}_ns{ns_exp_str}_lr{lr_start_str}_lrmin{lr_min_str}_d{vector_size}_w{window}_e{epochs}_hs{hs}_neg{negative}_mincount{min_count}"


def _model_train(
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

    """Train Word2Vec model.

    Args:  
        corpus_path (str): Path to corpus
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

    Returns:  
        model (wv.Word2Vec): Word2Vec model

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
        model.train(corpus_iterable=sentences, 
            total_examples=model.corpus_count, 
            epochs=model.epochs,
            compute_loss=True,
            callbacks=callbacks)
        print(model.__dict__)
        bar()
        
    return model


def _umap_reduce(
    vectors_obj: wv.KeyedVectors, 
    n_dims: int):

    """ Reduce dimensionality of vectors using UMAP of embedded vectors.

    Args:  
        vectors_obj (wv.KeyedVectors): Word2Vec model.
        n_dims (int): Number of dimensions to reduce to.

    Returns: 
        embedding (np.ndarray): Reduced dimensionality vectors.

    """

    with alive_bar(title = "UMAP Magic",  dual_line = True, spinner = PHROG_SPINNER) as bar:
        reducer = umap.UMAP(n_components=n_dims)
        data_to_reduce = vectors_obj.vectors
        # reduce dimensionality
        embedding = reducer.fit_transform(data_to_reduce)
        bar()
    return embedding


def _model_visualise(vectors_obj: wv.KeyedVectors, 
                    reduced_embed: np.ndarray, 
                    visual_path: str,
                    encoded: bool):

    """ Visualize model using UMAP reduced vectors in 3D by plotting with Plotly.

    Args:  
        vectors_obj (wv.KeyedVectors): Word2Vec model.
        reduced_embed (np.ndarray): Reduced dimensionality vectors.
        visual_path (str): Path to save visualization.
        encoded (bool): Whether the model is encoded or not.

    Returns: 
        None.

    """

    with alive_bar(title = "Gathering phrog metadata and embedding data",  dual_line = True, spinner = PHROG_SPINNER) as bar:
        dataset = pd.DataFrame({'word': vectors_obj.index_to_key})
        if not encoded:
            func = utils.read_metadata(Path("Data/metadata_phrog.pickle"))
        else:
            func = utils.read_metadata(Path("Data/metadata_phrog_coded.pickle"))
        dataset["function"] = dataset['word'].map(func)
        dataset[['x', 'y', 'z']] = pd.DataFrame(reduced_embed, index=dataset.index)
        bar()
    
    with alive_bar(title = "Generating visualisation",  dual_line = True, spinner = PHROG_SPINNER) as bar:
        fig = px.scatter_3d(dataset, x='x', y='y', z='z', color='function', hover_data=["word"], color_discrete_map=utils.colour_map)
        fig.update_traces(marker_size = 4)
        fig.write_html(Path(visual_path).as_posix())
        bar()

