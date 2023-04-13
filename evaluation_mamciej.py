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
    

@utils.time_this
def parallel_validation(func_dict_df, phrog_categories):
    score_tally = mp.Manager().dict()
    # function_tally = mp.Manager().dict()
    function_tally = defaultdict(mp.Manager().dict())
    mini_function_tally = defaultdict(lambda: defaultdict(int))
    # mini_function_tally = {}
    processes = []

    for phrog, scoring_functions in phrog_categories.items():
        for scoring_function, assigned_category in scoring_functions.items():
            if scoring_function not in mini_function_tally.keys():
                mini_function_tally[scoring_function] = {}
                mini_function_tally[scoring_function][assigned_category[0]] = 0
            if assigned_category[0] not in mini_function_tally[scoring_function].keys():
                mini_function_tally[scoring_function][assigned_category[0]] = 0

    print(mini_function_tally)

    for key in mini_function_tally:
        function_tally[key] = mp.Manager().dict()

    # Divide phrog_categories into chunks
    num_chunks = mp.cpu_count()
    chunk_size = int(len(phrog_categories) / num_chunks) + 1
    chunks = [dict(list(phrog_categories.items())[i:i+chunk_size])
              for i in range(0, len(phrog_categories), chunk_size)]

    # Start a process for each chunk
    for chunk in chunks:
        process = mp.Process(target=validate_chunk, args=(
            func_dict_df, chunk, score_tally, function_tally, mini_function_tally))
        process.start()
        processes.append(process)
    # Wait for all processes to finish
    for process in processes:
        process.join()

    # Convert answer_tally to a regular dictionary and calculate percentages
    score_tally = dict(score_tally)
    for scoring_function, n_true_answers in score_tally.items():
        score_tally[scoring_function] = round(
            (n_true_answers / len(phrog_categories)) * 100, 2)

    function_tally = dict(function_tally)

    print ('\n\n I-- CHECK THOSE NUMBERS --I')  # TODO check if this works correctly (new changes with function_tally)
    print('Correctly assigned raw count:', function_tally)
    function_counts = {}
    for category in func_dict_df['category'].values.tolist():
        function_counts[category] = function_counts.get(category, 0) + 1
    print('Total count:', function_counts)
    for scoring_function in function_tally:
        for category in function_tally[scoring_function]:
            function_tally[scoring_function][category] = round((function_tally[scoring_function][category] / function_counts[category]) * 100, 2)

    return score_tally, function_tally


def validate_chunk(func_dict_df, phrog_categories, score_tally, function_tally, mini_function_tally):
    local_score_tally = {}
    # local_function_tally = {}
    local_function_tally = defaultdict(lambda: defaultdict(int))


    for phrog, scoring_functions in phrog_categories.items():
        true_category = func_dict_df.loc[func_dict_df['phrog_id'] == phrog, 'category'].values[
            0]  # get the proper category of the phrog
        for scoring_function, assigned_category in scoring_functions.items():
            if scoring_function not in local_score_tally.keys():
                local_score_tally[scoring_function] = 0
            if scoring_function not in local_function_tally.keys():
                local_function_tally[scoring_function] = {}
                local_function_tally[scoring_function][assigned_category[0]] = 0
            if assigned_category[0] not in local_function_tally[scoring_function].keys():
                local_function_tally[scoring_function][assigned_category[0]] = 0
            if assigned_category[0] == true_category:
                local_score_tally[scoring_function] += 1
                local_function_tally[scoring_function][true_category] += 1

    # Update the shared answer_tally dictionary atomically
    for scoring_function, count in local_score_tally.items():
        with mp.Lock():
            score_tally[scoring_function] = score_tally.get(
                scoring_function, 0) + count

    # for scoring_function, count in local_function_tally.items():
    #     with mp.Lock():
    #         for phrog_category, value in count.items():
    #             print(scoring_function, value, phrog_category)
    #             mini_function_tally[scoring_function][phrog_category] += value
    #         function_tally[scoring_function] = mini_function_tally[scoring_function]
    for scoring_function, count in local_function_tally.items():
        with mp.Lock():
            for phrog_category, value in count.items():
                print(scoring_function, value, phrog_category)
                mini_function_tally[scoring_function][phrog_category] += value
            function_tally[scoring_function] = dict(mini_function_tally[scoring_function])


# @utils.time_this
def batch_exec(phrog_batch, vectors, func_dict_df, top_known_phrogs):
    local_phrog_categories: dict[str, dict[str, str]] = {}
    print(len(phrog_batch))
    for phrog in phrog_batch:
        # start = time.perf_counter()
        try:
            result = vectors.most_similar(phrog, topn=60_000)
        except KeyError:
            continue

         # replace phrogs with functions
        model_result_tuples_to_df = pd.DataFrame(
            result, columns=['phrog_id', 'probability'])
        merged = model_result_tuples_to_df.merge(
            func_dict_df, on='phrog_id')  # Same col, just use on=
        # just use loc, add .head immeadiately
        # separated to accomodate a very rare edge 
        merged = merged.loc[((merged['category'] != 'unknown function') & (
        merged['category'] != 'other')& (merged['category'] != 'moron, auxiliary metabolic gene and host takeover'))]
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
        # end = time.perf_counter()
        # runtime = end - start
        # print(f"Done one iteration of phrog from one frog batch in {runtime:0.8f}")
    return local_phrog_categories


def batch_list(item_list, batch_count: int = cpu_count() - 1):
    batches = np.array_split(np.array(item_list), batch_count)
    return batches


# @utils.time_this
def parallel_scoring(phrog, merged_id_category):
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


def prediction(func_dict: dict, model: Union[FastText, Word2Vec], 
               model_name: str, top_known_phrogs: int = 50, evaluate_mode: bool = True):
    
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
        batch, vectors, func_dict_df, top_known_phrogs) for batch in alive_it(batch_list(phrogs_to_predict), dual_line=True, spinner=PHROG_SPINNER))

    start = time.perf_counter()
    phrog_categories = {
        k: v for x in list_phrog_categories for k, v in x.items()}
    end = time.perf_counter()
    runtime = end - start
    print(f"Done phrog_categories in {runtime:0.8f}")
    # print(phrog_categories)

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
        scores, func_scores = parallel_validation(func_dict_df, phrog_categories)
        print('Correctly assigned procentage:', func_scores)
        max_value = max(scores.values())  # maximum value
        max_scoring_func = [k for k, v in scores.items() if v == max_value]
        print(f"{max_value}%")
        char_nl = '\n'
        with open("evaluation_log.txt", "a") as f:  # very rudimentary logging as of now
            f.write(f"{type(model).__name__}/{model_name}{str(scores)}{char_nl}")
        return scores

