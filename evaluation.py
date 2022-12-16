from pathlib import Path
import pandas as pd
from typing import Union
from gensim.models import FastText, Word2Vec
from collections import defaultdict
import custom_logger

import utils


def sum_tuples(lst: list[str, float]):
    d = defaultdict(float)
    for category, prob in lst:
        d[category] += prob

    return list(d.items())


def mean_tuples(lst: list[str, float]):
    d = defaultdict(float)
    occurs = defaultdict(int)
    for category, prob in lst:
        d[category] += prob
        occurs[category] += 1
    for category in d:
        d[category] /= occurs[category]
    return list(d.items())


def power_tuples(lst: list[str, float], power):
    for cat_prob_list in lst:
        cat_prob_list[1] = cat_prob_list[1]**power

    return lst


@utils.time_this
def prediction(
        func_dict: dict,
        model: Union[FastText, Word2Vec],
        top_known_phrogs: int
):
    # convert dict to pandas dataframe or read it directly
    func_dict_df = pd.DataFrame(func_dict.items(), columns=['phrog_id', 'category'])
    # df = pd.read_table('Data/metadata-phrog.tsv', header=0)

    # create a list of phrogs with known function
    known_func_phrog_list = func_dict_df[(func_dict_df['category'] != 'unknown function')]['phrog_id'].tolist()
    # TODO: Consider making this a set, O(1) membership checking
    # known_func_phrog_list = set(known_func_phrog_list)

    # select matches with known function
    # for phrog in known_func_phrog_list:   # <-- TODO: consider changing it to multiprocessing

    # Dictionary phrog: {
    #   scoring_function: category
    # }
    # Example:
    # 'phrog13': {
    #   'max': ('category1', 0.923),
    #   'sum': ('category3', 0.987),
    #   'mean': ('category1', 0.956)
    # }
    phrog_categories: dict[str, dict[str, str]] = {}

    for phrog in known_func_phrog_list[
                 :10]:  # placeholder, model wasn't trained for all phrogs yet, tested to work on phrogs1-10
        vectors = model.wv
        result = vectors.most_similar(phrog,
                                      topn=40_000)  # Use _ for clarity # <-- TODO: looking for ideas how to optimize this
        # result = (list(filter(lambda x: 'joker' not in x[0], result)))  # to remove jokers from result; turns out mergeddf_to_tuple isnt returning them anyway so far

        # replace phrogs with functions
        model_result_tuples_to_df = pd.DataFrame(result, columns=['phrog_id', 'probability'])
        merged = model_result_tuples_to_df.merge(func_dict_df, on='phrog_id')  # Same col, just use on=

        # just use loc, add .head immeadiately
        # separated to accomodate a very rare edge case
        merged = merged.loc[merged['category'] != 'unknown function']
        if merged.empty:
            custom_logger.logger.error("All closest phrogs had unknown function - all were dropped, no data left to score.")
        merged = merged.head(top_known_phrogs)
        if len(merged) < top_known_phrogs:
            custom_logger.logger.warning("Not enough close phrogs with known function - "
                                         "scoring using less than {} phrogs.".format(top_known_phrogs))
        merged_id_category = merged[["category", "probability"]]

        # Begin scoring
        list_for_scoring = list(merged_id_category.apply(list, 1))
        key_func = lambda x: x[1]

        # 4 scoring functions
        # mx: max value for a category
        # sum: max value after summing prob for each category
        # mean: max value after taking a mean prob for each category
        # power: max value after summing probs to the power of n
        mx = tuple(max(list_for_scoring, key=key_func))#other functions return tuples, so...
        summed = max(sum_tuples(list_for_scoring), key=key_func)
        mean = max(mean_tuples(list_for_scoring), key=key_func)
        power_2 = max(sum_tuples(power_tuples(list_for_scoring, 2)), key=key_func)
        power_4 = max(sum_tuples(power_tuples(list_for_scoring, 4)), key=key_func)
        power_8 = max(sum_tuples(power_tuples(list_for_scoring, 8)), key=key_func)

        phrog_categories[phrog] = {
            "max": mx,
            "sum": summed,
            "mean": mean,
            "power 2": power_2,
            "power 4": power_4,
            "power 8": power_8
        }
        # End scoring
        # print(phrog_categories)

    # TODO: validation


#
######## --------------------------------- ########
# Temporary code for running as a separate script #  VVV

# read phrog:function dictionary
func = utils.read_metadata(Path("Data/metadata_phrog.dill"))
# read trained model
mod = Word2Vec.load('train_test/test.model')
# mod = FastText.load()

# run
prediction(func, mod, top_known_phrogs=50)
