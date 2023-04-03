from pathlib import Path
import pandas as pd
import numpy as np
import re
from typing import Union
from gensim.models import FastText, Word2Vec
# from collections import defaultdict
# import collections
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import codon

import custom_logger
import utils


# @codon.jit
def sum_tuples(lst: list[str, float]):
    # d = collections.defaultdict(float)
    d = {}
    print(lst[0:10])
    for category, prob in lst:
        d[category] += prob
    return list(d.items())


# @codon.jit
def mean_tuples(lst: list[str, float]):
    # d = collections.defaultdict(float)
    # occurs = collections.defaultdict(int)
    d = {}
    occurs = {}
    for category, prob in lst:
        d[category] += prob
        occurs[category] += 1
    for category in d:
        d[category] /= occurs[category]
    return list(d.items())


# @codon.jit
def power_tuples(lst: list[str, float], power):
    for cat_prob_list in lst:
        cat_prob_list[1] = cat_prob_list[1] ** power
    return lst


@utils.time_this
def validation(func_dict_df, phrog_categories):
    answer_tally = {}
    for phrog, scoring_functions in phrog_categories.items():
        true_category = func_dict_df.loc[func_dict_df['phrog_id'] == phrog, 'category'].values[
            0]  # get the proper category of the phrog
        for scoring_function, assigned_category in scoring_functions.items():
            if scoring_function not in answer_tally.keys():
                answer_tally[scoring_function] = 0
            if assigned_category[0] == true_category:
                answer_tally[scoring_function] = answer_tally[scoring_function] + 1
    for scoring_function, n_true_answers in answer_tally.items():
        answer_tally[scoring_function] = round(
            (n_true_answers / len(phrog_categories)) * 100, 2)
    return answer_tally


def batch_exec(phrog_batch, vectors, func_dict_df, top_known_phrogs):
    local_phrog_categories: dict[str, dict[str, str]] = {}
    print(len(phrog_batch))
    for phrog in phrog_batch:
        try:
            result = vectors.most_similar(phrog, topn=60_000)
            # result = (list(filter(lambda x: 'joker' not in x[0], result)))  # to remove jokers from result; turns out mergeddf_to_tuple isnt returning them anyway so far

        except KeyError:
            continue

         # replace phrogs with functions
        model_result_tuples_to_df = pd.DataFrame(
            result, columns=['phrog_id', 'probability'])
        merged = model_result_tuples_to_df.merge(
            func_dict_df, on='phrog_id')  # Same col, just use on=
        # just use loc, add .head immeadiately
        # separated to accomodate a very rare edge case
        merged = merged.loc[merged['category'] != 'unknown function']
        if merged.empty:
            custom_logger.logger.error("All closest phrogs had unknown function - "
                                       "all were dropped, no data left to score.")
        merged = merged.head(top_known_phrogs)
        if len(merged) < top_known_phrogs:
            custom_logger.logger.warning("Not enough close phrogs with known function - "
                                         "scoring using less than {} phrogs.".format(top_known_phrogs))
        merged_id_category = merged[["category", "probability"]]
        local_phrog_categories.update(
            parallel_scoring(phrog, merged_id_category))
    return local_phrog_categories


def batch_list(item_list, batch_count: int = cpu_count() - 1):
    batches = np.array_split(np.array(item_list), batch_count)
    return batches


def parallel_scoring(phrog, merged_id_category):
    d_phrog_categories = {}
    list_for_scoring = list(merged_id_category.apply(list, 1))
    def key_func(x): return x[1]

    # 4 scoring functions
    # mx: max value for a category
    # sum: max value after summing prob for each category
    # mean: max value after taking a mean prob for each category
    # power: max value after summing probs to the power of n
    # other functions return tuples, so...
    mx = tuple(max(list_for_scoring, key=key_func))
    summed = max(sum_tuples(list_for_scoring), key=key_func)
    mean = max(mean_tuples(list_for_scoring), key=key_func)
    power_2 = max(sum_tuples(power_tuples(list_for_scoring, 2)), key=key_func)
    power_4 = max(sum_tuples(power_tuples(list_for_scoring, 4)), key=key_func)
    power_8 = max(sum_tuples(power_tuples(list_for_scoring, 8)), key=key_func)

    d_phrog_categories[phrog] = {
        "max": mx,
        "sum": summed,
        "mean": mean,
        "power 2": power_2,
        "power 4": power_4,
        "power 8": power_8
    }

    return d_phrog_categories


def prediction(
    func_dict: dict,
    model: Union[FastText, Word2Vec],
    model_name: str,
    top_known_phrogs: int = 50,
    evaluate_mode: bool = True,
):
    # convert dict to pandas dataframe or read it directly
    func_dict_df = pd.DataFrame(func_dict.items(), columns=[
                                'phrog_id', 'category'])
    # df = pd.read_table('Data/metadata-phrog.tsv', header=0)

    # create a list of phrogs with known function
    known_func_phrog_list = func_dict_df[((func_dict_df['category'] != 'unknown function') & (
        func_dict_df['category'] != 'other'))]['phrog_id'].tolist()
    # TODO: Consider making this a set, O(1) membership checking
    # known_func_phrog_list = set(known_func_phrog_list)

    vectors = model.wv

    if evaluate_mode:
        phrogs_to_predict = known_func_phrog_list
    else:
        model_keys = model.wv.key_to_index
        phrogs_to_predict = [
            w for w in model_keys if not w[-1].isdigit() or 'joker' in w]

    # parallel function to select best matches and score the model
    print(len(phrogs_to_predict))
    list_phrog_categories = Parallel(verbose=True, n_jobs=-1)(delayed(batch_exec)(
        batch, vectors, func_dict_df, top_known_phrogs) for batch in batch_list(phrogs_to_predict))
    # transform list of dicts to dict
    phrog_categories = {
        k: v for x in list_phrog_categories for k, v in x.items()}
    # print(phrog_categories)

    # Dictionary phrog: {
    #   scoring_function: category
    # }
    # Example:
    # 'phrog13': {
    #   'max': ('category1', 0.923),
    #   'sum': ('category3', 0.987),
    #   'mean': ('category1', 0.956)
    # }

    # validation

    if evaluate_mode:
        scores = validation(func_dict_df, phrog_categories)
        max_value = max(scores.values())  # maximum value
        max_scoring_func = [k for k, v in scores.items() if v == max_value]
        print("{}%".format(max_value))
        char_nl = '\n'
        with open("evaluation_log.txt", "a") as f:  # very rudimentary logging as of now
            f.write(f"{type(model).__name__}/{model_name}{str(scores)}{char_nl}")
        return scores

#
######## --------------------------------- ########
# # Temporary code for running as a separate script #  VVV

# # read phrog:function dictionary
# func = utils.read_metadata(Path("Data/metadata_filtered.pickle"))

# # read trained model
# mod =  Word2Vec.load('XYZ')
# mod = FastText.load('train_test/virall_encode_better_CODED_31-12_ft_ns-075_lr0005_lrmin00001_d25_w2_e150_hs0_neg75_maxn3_minn6.model')

# # run
# prediction(func, mod)
