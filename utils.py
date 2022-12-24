import time
import pandas as pd
from pathlib import Path
import pickle
import custom_logger


colour_map = {
    'moron, auxiliary metabolic gene and host takeover': '#636EFA',
    'tail': '#EF553B',
    'DNA, RNA and nucleotide metabolism': '#00CC96',
    'other': '#AB63FA',
    'unknown function': '#FFA15A',
    'head and packaging': '#19D3F3',
    'integration and excision': '#FF6692',
    'connector': '#B6E880', 
    'transcription regulation': '#FF97FF', 
    'lysis': '#FECB52'
}


def metadata_dump():
    """
    Dump metadata as dictionary ("phrog_id": "funcion") to a dill file
    """

    df = pd.read_table('Data/metadata-phrog.tsv', header=0)
    metadata_dict = dict(zip(df['phrog_id'], df['category']))
    pickle.dump(metadata_dict, open('Data/metadata_phrog.pickle', 'wb'))


def read_corpus(path: Path) -> 'list[list]':
    """
    Reads and returns corpus from given corpus file
    """

    custom_logger.logger.info("Loading pickle with corpus")

    try:
        with open(path.as_posix(), 'rb') as f:
            sentences = pickle.load(f)
        return sentences
    except (pickle.UnpicklingError, FileNotFoundError):
        custom_logger.logger.critical("Incorrect or corrupted file!")
        return


def read_metadata(path: Path) -> pd.DataFrame:
    """
    Reads and returns metadata pickle from given metadata pickle file
    """
    custom_logger.logger.info("Loading dill with phrog metadata")

    try:
        with open(path.as_posix(), 'rb') as in_strm:
            func = pickle.load(in_strm)
            return func
    except (pickle.UnpicklingError, FileNotFoundError):
        custom_logger.logger.critical("Incorrect or corrupted file!")
        return


def time_this(function):
    """
    Decorator to return function execution time.
    """
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        exec = function(*args, **kwargs)
        end = time.perf_counter()
        runtime = end - start
        custom_logger.logger.info(f"{function.__name__} >>> Done in {runtime:0.8f}")
        return exec

    return wrapper


def create_phrog_lookup(path: Path):
    try:
        with open(path.as_posix(), 'rb') as in_strm:
            corpus = pickle.load(in_strm)
            print(corpus[0][0][-5:])
            # lookup = [x for l in [elem for elem in corpus if not elem.endswith('#####')] for x in l]
            lookup = {x: f"phrog_{str(int(x[-5:]))}" for elem in corpus for x in elem if not x.endswith("#####")}
            # print(lookup)
        with open("Data/metadata_lookup_phrog.pickle", 'wb') as fh:
            pickle.dump(lookup, fh)
    except (pickle.UnpicklingError, FileNotFoundError):
        custom_logger.logger.critical("Incorrect or corrupted file!")
        return


def read_lookup_metadata(path: Path) -> dict:
    """
    Reads and returns metadata pickle from given metadata pickle file
    """
    custom_logger.logger.info("Loading dill with phrog metadata")

    try:
        with open(path.as_posix(), 'rb') as in_strm:
            func = pickle.load(in_strm)
            return func
    except (pickle.UnpicklingError, FileNotFoundError):
        custom_logger.logger.critical("Incorrect or corrupted file!")
        return