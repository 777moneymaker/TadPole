from pathlib import Path
import pandas as pd
import numpy as np
from typing import Union
from gensim.models import FastText, Word2Vec
from collections import defaultdict
from joblib import Parallel, delayed, parallel_backend
from multiprocessing import cpu_count
import multiprocessing as mp
import time
from alive_progress import alive_it
from alive_progress.animations.spinners import bouncing_spinner_factory
from tqdm import tqdm
from typing import Tuple

import custom_logger
import utils


PHROG_SPINNER = bouncing_spinner_factory(
    ("🐸", "🐸"), 8, background=".", hide=False, overlay=True)


def sum_tuples(lst):
    d = defaultdict(float)
    for category, prob in lst:
        d[category] += prob
    return list(d.items())


def mean_tuples(lst):
    d = defaultdict(float)
    occurs = defaultdict(int)
    for category, prob in lst:
        d[category] += prob
        occurs[category] += 1
    return [(category, d[category] / occurs[category]) for category in d]


def power_tuples(lst, power):
    return [(cat_prob_list[0], cat_prob_list[1] ** power) for cat_prob_list in lst]
    

# Divide phrog_categories into chunks
def batch_dict(phrog_categories, num_chunks: int = cpu_count()):
    chunk_size = int(len(phrog_categories) / num_chunks) + 1
    chunks = [dict(list(phrog_categories.items())[i:i+chunk_size])
              for i in range(0, len(phrog_categories), chunk_size)]
    return chunks

def validate_chunk(func_dict_df, phrog_categories):
    local_score_tally = {}
    local_function_tally = {}
    local_used_phrog_function_tally = {}
    local_used_phrog_name_tally = {}
    for phrog, scoring_functions in phrog_categories.items():
        true_category = func_dict_df.loc[func_dict_df['phrog_id'] == phrog, 'category'].values[
            0]  # get the proper category of the phrog
        for scoring_function, assigned_category in scoring_functions.items():
            if scoring_function not in local_score_tally.keys():
                local_score_tally[scoring_function] = 0
            if (scoring_function,assigned_category[0]) not in local_function_tally.keys():
                local_function_tally[(scoring_function,assigned_category[0])] = 0
            if assigned_category[0] == true_category:
                local_score_tally[scoring_function] += 1
                local_function_tally[(scoring_function,true_category)] += 1
        true_category = func_dict_df.loc[func_dict_df['phrog_id'] == phrog, 'category'].values[0]  # get the proper category of the phrog
        if true_category not in local_used_phrog_function_tally.keys():
            local_used_phrog_function_tally[true_category] = 0
        local_used_phrog_function_tally[true_category] += 1
    return local_score_tally, local_function_tally, local_used_phrog_function_tally

# @utils.time_this
def batch_exec(phrog_batch, vectors, func_dict_df, top_known_phrogs, power_range):
    local_phrog_categories: dict[str, dict[str, str]] = {}
    # print(len(phrog_batch))
    for phrog in tqdm(phrog_batch):
        # start = time.perf_counter()
        try:
            result = [vector for vector in vectors.most_similar(phrog, topn=60_000) if not vector[0].endswith(phrog[-5:])]
        except KeyError:
            continue

        # replace phrogs with functions
        model_result_tuples_to_df = pd.DataFrame(
            result, columns=['phrog_id', 'probability'])
        model_result_tuples_to_df.astype({'phrog_id': 'string[pyarrow]', 'probability': 'float[pyarrow]'})
        merged = model_result_tuples_to_df.merge(
            func_dict_df, on='phrog_id')  # Same col, just use on=
        # just use loc, add .head immeadiately
        # separated to accomodate a very rare edge 
        merged = merged.loc[((merged['category'] != 'unknown function') & (
        merged['category'] != 'other') & (merged['category'] != 'moron, auxiliary metabolic gene and host takeover'))]
        if merged.empty:
            custom_logger.logger.error("All closest phrogs had unknown function - "
                                       "all were dropped, no data left to score.")
        merged = merged.head(top_known_phrogs)
        if len(merged) < top_known_phrogs:
            custom_logger.logger.warning("Not enough close phrogs with known function - "
                                         "scoring using less than {} phrogs.".format(top_known_phrogs))
        merged_id_category = merged[["category", "probability"]]
        local_phrog_categories.update(
            parallel_scoring(phrog, merged_id_category, power_range))
        # end = time.perf_counter()
        # runtime = end - start
        # print(f"Done one iteration of phrog from one frog batch in {runtime:0.8f}")
    return local_phrog_categories


def batch_list(item_list, batch_count: int = cpu_count() - 1):
    batches = np.array_split(np.array(item_list), batch_count)
    return batches


# @utils.time_this
def parallel_scoring(phrog, merged_id_category, power_range):
    d_phrog_categories = {}
    list_for_scoring = [list(row)
                        for row in merged_id_category.itertuples(index=False)]

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
    # power_3 = max(sum_tuples(power_tuples(list_for_scoring, 3)), key=key_func)
    # power_4 = max(sum_tuples(power_tuples(list_for_scoring, 4)), key=key_func)
    # power_5 = max(sum_tuples(power_tuples(list_for_scoring, 5)), key=key_func)

    d_phrog_categories[phrog] = {
        "max": mx,
        "sum": summed,
        "mean": mean,
        # "power 3": power_3,
        # "power 4": power_4,
        # "power 5": power_5
    }

    powers = [(f"power {power}", max(sum_tuples(power_tuples(list_for_scoring, power)), key=key_func)) for power in np.arange(*power_range)]
    d_phrog_categories[phrog].update(powers)

    return d_phrog_categories


def prediction(func_dict: dict, model: Union[FastText, Word2Vec], 
               model_name: str, top_known_phrogs: int = 50, 
               evaluate_mode: bool = True,
               power_range: Tuple[float, float, float] = (3, 5.2, 0.2)):
    
    # convert dict to pandas dataframe or read it directly
    start = time.perf_counter()
    func_dict_df = pd.DataFrame(func_dict.items(), columns=['phrog_id', 'category'])
    end = time.perf_counter()
    runtime = end - start
    print(f"Done func_dict_df in {runtime:0.8f}")

    # create a list of phrogs with known function
    start = time.perf_counter()
    known_func_phrog_list = func_dict_df[((func_dict_df['category'] != 'unknown function') & (
        func_dict_df['category'] != 'other')& (func_dict_df['category'] != 'moron, auxiliary metabolic gene and host takeover'))]['phrog_id'].tolist()
    end = time.perf_counter()
    runtime = end - start
    print(f"Done known_func_phrog_list in {runtime:0.8f}")

    vectors = model.wv
    if evaluate_mode:
        phrogs_to_predict = known_func_phrog_list
    else:
        model_keys = model.wv.key_to_index
        phrogs_to_predict = [w for w in model_keys if not w[-1].isdigit() or 'joker' in w]

    # parallel function to select best matches and score the model
    print(len(phrogs_to_predict))
    list_phrog_categories = Parallel(verbose=True, n_jobs=-1)(delayed(batch_exec)(
        batch, vectors, func_dict_df, top_known_phrogs, power_range) for batch in alive_it(batch_list(phrogs_to_predict), dual_line=True, spinner=PHROG_SPINNER))

    start = time.perf_counter()
    phrog_categories = {
        k: v for x in list_phrog_categories for k, v in x.items()}
    end = time.perf_counter()
    print(len(phrog_categories))
    runtime = end - start
    print(f"Done phrog_categories in {runtime:0.8f}")

    # TODO: put to docstring
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
        result = Parallel(verbose=True, n_jobs=-1)(delayed(validate_chunk)(func_dict_df, chunk) for chunk in batch_dict(phrog_categories))
        print(result)
        score_tally = defaultdict(int)
        function_tally = defaultdict(int)
        used_phrog_function_tally = defaultdict(int)
        for tup in result:
            for elem in tup[0]:
                score_tally[elem] += tup[0][elem]
            for elem in tup[1]:
                function_tally[elem] += tup[1][elem]
            for elem in tup[2]:
                used_phrog_function_tally[elem] += tup[2][elem]
        score_tally = dict(score_tally)
        function_tally = dict(function_tally)
        used_phrog_function_tally = dict(used_phrog_function_tally)
        # print(f"used_phrog_function_tally: {used_phrog_function_tally}")
        # print(f"function_tally: {function_tally}")
        # print(f"score_tally: {score_tally}")
        for scoring_function, n_true_answers in score_tally.items():
            score_tally[scoring_function] = round((n_true_answers / len(phrog_categories)) * 100, 2)
        for category in function_tally:
            function_tally[category] = round((function_tally[category] / used_phrog_function_tally[category[1]]) * 100, 2)
        scores = score_tally
        func_scores = function_tally
        max_value = max(scores.values())  # maximum value
        max_scoring_func = [k for k, v in scores.items() if v == max_value]
        max_func_scores = {}
        for key, value in func_scores.items():
            if key[0] == max_scoring_func[0]:
                max_func_scores[key[1]] = value
        print('Correctly assigned procentages: ', max_func_scores)
        print(f"{max_value}%")
        char_nl = '\n'
        with open("evaluation_log.txt", "a") as f:  # very rudimentary logging as of now
            f.write(f"{type(model).__name__}/{model_name}{str(scores)}{char_nl}")
        not_evaluated_num = len(phrogs_to_predict) - len(phrog_categories)
        return scores, max_func_scores, not_evaluated_num
