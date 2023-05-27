import time
import pandas as pd
from pathlib import Path
import pickle
import custom_logger
import pymsteams
from copy import deepcopy
from typing import Dict, Set, Union
from gensim.models import Word2Vec, FastText, KeyedVectors


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


def create_function_lookup(path: Path, out_path: Path):
    try:
        with open(path.as_posix(), 'rb') as in_strm:
            corpus = pickle.load(in_strm)
            metadata = read_metadata(Path("Data/metadata_phrog.pickle"))
            lookup = {x: metadata[f"phrog_{str(int(x[-5:]))}"] for elem in corpus for x in elem if not x.endswith("#####")}
        # with open("Data/metadata_phrog_coded.pickle", 'wb') as fh:
        with open(out_path.as_posix(), 'wb') as fh:
            pickle.dump(lookup, fh)
    except:
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


def exchange_lookup(path: Path):
    org_lookup = read_lookup_metadata(path)
    new_lookup = dict((v, k) for k, v in org_lookup.items())
    # print(new_lookup)
    with open("Data/metadata_lookup_phrog_exchanged.pickle", 'wb') as fh:
            pickle.dump(new_lookup, fh)


def make_encoded_metadata(metadata_path: Path, lookup_path: Path):
    org_metadata = read_metadata(metadata_path)
    # print(org_metadata['phrog_1115'])
    lookup = read_lookup_metadata(lookup_path)
    # print(lookup['phrog_1115'])
    new_metadata = dict((lookup[k] if k in lookup else k, v) for k, v in org_metadata.items())
    # print(new_metadata['phrog_1115'])
    with open("Data/metadata_phrog_encoded.pickle", 'wb') as fh:
            pickle.dump(new_metadata, fh)


def filter_metadata(corpus_path: Path, metadata_path: Path, out_path: Path, coded: bool):
    corpus = read_corpus(corpus_path)
    org_metadata = read_metadata(metadata_path)
    if coded:
        corpus_phrogs = set((f"phrog_{str(int(x[-5:]))}" for elem in corpus for x in elem if not x.endswith("#####")))
    else:
        corpus_phrogs = set((x for elem in corpus for x in elem if x.startswith('phrog')))
    print(len(corpus_phrogs))
    all_phrogs = set(org_metadata.keys())
    print(len(all_phrogs))
    lost_phrogs = all_phrogs - corpus_phrogs
    print(len(lost_phrogs))
    filtered_metadata = {phrog: org_metadata[phrog] for phrog in corpus_phrogs}
    # filtered_metadata = {f"phrog_{str(int(x[-5:]))}": org_metadata[f"phrog_{str(int(x[-5:]))}"] for elem in corpus for x in elem if not x.endswith("#####")}
    with open(out_path.as_posix(), 'wb') as fh:
            pickle.dump(filtered_metadata, fh)
    return lost_phrogs


async def report_opt_result_teams(**kwargs):
    api_key = ""
    message = pymsteams.async_connectorcard(api_key)
    message.title("Optymalizacja zakończona")
    message.text("@Kanał Testowy\n trzeba fixa, ale wygenerowane przez optymalizator")
    message.color("34eb80")
    main_section = pymsteams.cardsection()
    main_section.enableMarkdown()
    main_section.title(f"# {kwargs.get('opt_name', 'Unassigned')}") # optname
    categories = "\r".join([f"- *{category_name}*: {category_value}" for category_name, category_value in kwargs['category'].items()])
    hypers = "\r".join([f"- *{hyper_name}*: {hyper_value}" for hyper_name, hyper_value in kwargs['hypers'].items()])
    # print(categories)
    # print(hypers)
    main_text = f"""

**Best:** {kwargs.get('best', 0)} ({kwargs.get('best_func', None)})

**Klasyfikacje kategorii:**

{categories}

**Hiperparametry:** 

{hypers}

    """
    main_section.enableMarkdown()
    main_section.text(main_text)
    message.addSection(main_section)
    
    await message.send()


def sanitze_vectors(model: Union[Word2Vec, FastText],
                    prefix: str = None,
                    filter: Set[str] = None,
                    translator_dict: Dict[str, str] = None) -> KeyedVectors:
    """

    :param model: trained Word2Vec or FastText model
    :param prefix: word prefix used to filter words e.g. "phrog_"
    :param filter: set of words to keep in the model
    :param translator_dict: {word_from_model: word_to_model} (e.g. {sequence: id})
    :return: filtered "model.wv" KeyedVectors object
    """
    sanitzed = deepcopy(model.wv)

    if prefix:
        tokens_to_keep = {k: i for k, i in sanitzed.key_to_index.items() if k.startswith(prefix)}
    elif filter:
        tokens_to_keep = {k: i for k, i in sanitzed.key_to_index.items() if k in filter}
    elif translator_dict:
        tokens_to_keep = {translator_dict[k]: i for k, i in sanitzed.key_to_index.items() if k in translator_dict}
    else:
        raise ValueError(f'Exactly one of the: "prefix", "filter_list", "translator_dict"arguments is required')
    indices = list(tokens_to_keep.values())
    sanitzed.key_to_index = {k: i for i, k in enumerate(tokens_to_keep)}
    sanitzed.index_to_key = [k for k in tokens_to_keep]
    sanitzed.vectors = sanitzed.vectors[indices]
    return sanitzed
