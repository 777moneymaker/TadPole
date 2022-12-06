import dill
import pandas as pd
from pathlib import Path
import pickle
import custom_logger


def metadata_dump():
    """
    Dump metadata as dictionary ("phrog_id": "funcion") to a dill file
    """

    df = pd.read_table('Data/metadata-phrog.tsv', header=0)
    metadata_dict = dict(zip(df['phrog_id'], df['category']))
    dill.dump(metadata_dict, open('Data/metadata_phrog.dill', 'wb'))


def read_corpus(path: Path) -> list[list]:
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
            func = dill.load(in_strm)
            return func
    except (dill.UnpicklingError, FileNotFoundError):
        custom_logger.logger.critical("Incorrect or corrupted file!")
        return
