from pathlib import Path
import pandas as pd
import numpy as np
import re
from typing import Union
from gensim.models import FastText, Word2Vec
from collections import defaultdict
# import collections
from joblib import Parallel, delayed, parallel_backend
from multiprocessing import cpu_count
import multiprocessing as mp
# import codon
import concurrent.futures
import time
from alive_progress import alive_it
# from alive_progress import alive_bar
from alive_progress.animations.spinners import bouncing_spinner_factory

import custom_logger
import utils


PHROG_SPINNER = bouncing_spinner_factory(("🐸", "🐸"), 8, background = ".", hide = False, overlay =True)

# @codon.jit
def sum_tuples(lst):
    d = defaultdict(float)
    # d = {}
    for category, prob in lst:
        # d.setdefault(category, 0.0)
        d[category] += prob
    return list(d.items())


# @codon.jit
def mean_tuples(lst):
    d = defaultdict(float)
    occurs = defaultdict(int)
    # d = {}
    # occurs = {}
    for category, prob in lst:
        # d.setdefault(category, 0.0)
        # occurs.setdefault(category, 0)
        d[category] += prob
        occurs[category] += 1
    # conventional for loop
    # for category in d:
    #     d[category] /= occurs[category]
    # return list(d.items())
    return [(category, d[category] / occurs[category]) for category in d]


# @codon.jit
def power_tuples(lst, power):
    # conventional for-loop
    # for cat_prob_list in lst:
    #     cat_prob_list[1] = cat_prob_list[1] ** power
    # return lst
    return [(cat_prob_list[0], cat_prob_list[1] ** power) for cat_prob_list in lst]


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


# def compute_score(phrog, scoring_function):
#         true_category = func_dict_df.loc[func_dict_df['phrog_id'] == phrog, 'category'].values[0]
#         assigned_category = phrog_categories[phrog][scoring_function][0]
#         return 1 if assigned_category == true_category else 0


# @utils.time_this
# def parallel_validation(func_dict_df, phrog_categories, workers= cpu_count() - 1):
#     answer_tally = {}

#     with mp.Pool(processes=workers) as pool:
#         futures = []
#         for phrog, scoring_functions in phrog_categories.items():
#             for scoring_function in scoring_functions:
#                 future = pool.apply_async(compute_score, (phrog, scoring_function))
#                 futures.append(future)
    
#         for future in futures:
#             scoring_function = future.get()
#             if scoring_function not in answer_tally.keys():
#                 answer_tally[scoring_function] = 0
#             answer_tally[scoring_function] += scoring_function
    
#     for scoring_function, n_true_answers in answer_tally.items():
#         answer_tally[scoring_function] = round(
#             (n_true_answers / len(phrog_categories)) * 100, 2)
#     return answer_tally


@utils.time_this
def parallel_validation(func_dict_df, phrog_categories):
    answer_tally = mp.Manager().dict()
    processes = []

    # Divide phrog_categories into chunks
    num_chunks = mp.cpu_count()
    chunk_size = int(len(phrog_categories) / num_chunks) + 1
    chunks = [dict(list(phrog_categories.items())[i:i+chunk_size]) for i in range(0, len(phrog_categories), chunk_size)]

    # Start a process for each chunk
    for chunk in chunks:
        process = mp.Process(target=validate_chunk, args=(func_dict_df, chunk, answer_tally))
        process.start()
        processes.append(process)

    # Wait for all processes to finish
    for process in processes:
        process.join()

    # Convert answer_tally to a regular dictionary and calculate percentages
    answer_tally = dict(answer_tally)
    for scoring_function, n_true_answers in answer_tally.items():
        answer_tally[scoring_function] = round(
            (n_true_answers / len(phrog_categories)) * 100, 2)
    return answer_tally

def validate_chunk(func_dict_df, phrog_categories, answer_tally):
    local_answer_tally = {}
    for phrog, scoring_functions in phrog_categories.items():
        true_category = func_dict_df.loc[func_dict_df['phrog_id'] == phrog, 'category'].values[
            0]  # get the proper category of the phrog
        for scoring_function, assigned_category in scoring_functions.items():
            if scoring_function not in local_answer_tally.keys():
                local_answer_tally[scoring_function] = 0
            if assigned_category[0] == true_category:
                local_answer_tally[scoring_function] += 1

    # Update the shared answer_tally dictionary atomically
    for scoring_function, count in local_answer_tally.items():
        with mp.Lock():
            answer_tally[scoring_function] = answer_tally.get(scoring_function, 0) + count


# @utils.time_this
def batch_exec(phrog_batch, vectors, func_dict_df, top_known_phrogs):
    local_phrog_categories: dict[str, dict[str, str]] = {}
    print(len(phrog_batch))
    # for phrog in alive_it(phrog_batch, dual_line = True, spinner = PHROG_SPINNER):
    for phrog in phrog_batch:
        # start = time.perf_counter()
        try:
            # result = [vector for vector in vectors.most_similar(phrog, topn=60_000) if not vector[0].endswith(phrog[-5:])]
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
        # end = time.perf_counter()
        # runtime = end - start
        # print(f"Done one iteration of phrog from one frog batch in {runtime:0.8f}")
    return local_phrog_categories


def batch_exec2(phrogs_to_predict, vectors, func_dict_df, top_known_phrogs):
    phrog_categories = mp.Manager().dict()
    processes = []

    # divide input into chunks
    chunks = np.array_split(np.array(phrogs_to_predict), mp.cpu_count())

    # start process for each chunk
    for chunk in chunks:
        process = mp.Process(target=compute_predictions, args=(chunk, vectors, func_dict_df, top_known_phrogs, phrog_categories))
        process.start()
        processes.append(process)
    
    # wait for all processes to finish
    for process in processes:
        process.join()
    
    # transform to builtin dict 
    return dict(phrog_categories)


def batch_exec_thread(phrog_batch, vectors, func_dict_df, top_known_phrogs):
    local_phrog_categories: dict[str, dict[str, str]] = {}
    print(len(phrog_batch))
    with parallel_backend("loky", inner_max_num_threads=cpu_count()):
        results = Parallel()(delayed(process_phrog)(phrog, vectors, func_dict_df, top_known_phrogs) for phrog in phrog_batch)
    for result in results:
        local_phrog_categories.update(result)
    return local_phrog_categories
    



def compute_predictions(phrog_batch, vectors, func_dict_df, top_known_phrogs, phrog_categories):
    # local_phrog_categories: dict[str, dict[str, str]] = {}
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
        # local_phrog_categories.update(
        #     parallel_scoring(phrog, merged_id_category))
        parallel_scoring2(phrog, merged_id_category, phrog_categories)

        # scores = parallel_scoring(phrog, merged_id_category)
        # with phrog_categories.get_lock():
        #     for k,v in scores.items():
        #         if k in phrog_categories:
        #             phrog_categories[k].update(v)
        #         else:
        #             phrog_categories[k] = v
    
    # TODO: refactor names
    # for k, v in local_phrog_categories.items():
    #     with mp.Lock():
    #         # phrog_categories[k] = phrog_categories.get(k, {})
    #         phrog_categories[k] = local_phrog_categories[k]

def process_phrog(phrog, vectors, func_dict_df, top_known_phrogs):
    try:
        result = vectors.most_similar(phrog, topn=60_000)
    except KeyError:
        return {}

    # replace phrogs with functions
    model_result_tuples_to_df = pd.DataFrame(
        result, columns=['phrog_id', 'probability'])
    merged = model_result_tuples_to_df.merge(
        func_dict_df, on='phrog_id')
    merged = merged.loc[merged['category'] != 'unknown function']
    if merged.empty:
        custom_logger.logger.error("All closest phrogs had unknown function - "
                                   "all were dropped, no data left to score.")
    merged = merged.head(top_known_phrogs)
    if len(merged) < top_known_phrogs:
        custom_logger.logger.warning("Not enough close phrogs with known function - "
                                     "scoring using less than {} phrogs.".format(top_known_phrogs))
    merged_id_category = merged[["category", "probability"]]
    return parallel_scoring(phrog, merged_id_category)


def batch_list(item_list, batch_count: int = cpu_count() - 1):
    batches = np.array_split(np.array(item_list), batch_count)
    return batches


def parallel_scoring2(phrog, merged_id_category, phrog_categories):
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

    phrog_categories[phrog] = {
        "max": mx,
        "sum": summed,
        "mean": mean,
        "power 2": power_2,
        "power 4": power_4,
        "power 8": power_8
    }

    return d_phrog_categories


# @utils.time_this
def parallel_scoring(phrog, merged_id_category):
    d_phrog_categories = {}
    # list_for_scoring = list(merged_id_category.apply(list, 1))    
    list_for_scoring = [list(row) for row in merged_id_category.itertuples(index=False)]
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
    start = time.perf_counter()
    func_dict_df = pd.DataFrame(func_dict.items(), columns=[
                                'phrog_id', 'category'])
    end = time.perf_counter()
    runtime = end - start
    print(f"Done func_dict_df in {runtime:0.8f}")
    # df = pd.read_table('Data/metadata-phrog.tsv', header=0)

    # create a list of phrogs with known function
    start = time.perf_counter()
    known_func_phrog_list = func_dict_df[((func_dict_df['category'] != 'unknown function') & (
        func_dict_df['category'] != 'other'))]['phrog_id'].tolist()
    # known_func_phrog_list = set(func_dict_df[~func_dict_df['category'].isin(['unknown function', 'other'])]['phrog_id'].tolist())
    end = time.perf_counter()
    runtime = end - start
    print(f"Done known_func_phrog_list in {runtime:0.8f}")
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
    # with alive_bar(title = "Evaluating",  dual_line = True, spinner = PHROG_SPINNER) as bar:
    # with parallel_backend("loky", inner_max_num_threads=cpu_count):
    list_phrog_categories = Parallel(prefer="threading", verbose=True, n_jobs=-1)(delayed(batch_exec)(
        batch, vectors, func_dict_df, top_known_phrogs) for batch in batch_list(phrogs_to_predict))
        # bar()

    # start = time.perf_counter()
    # # parallel using futures
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     futures = []
    #     for batch in batch_list(phrogs_to_predict):
    #         futures.append(executor.submit(batch_exec, batch, vectors, func_dict_df, top_known_phrogs))
        
    #     list_phrog_categories = []
    #     for future in concurrent.futures.as_completed(futures):
    #         list_phrog_categories.append(future.result())
    # end = time.perf_counter()
    # runtime = end - start
    # print(f"Done parallel futures in {runtime:0.8f}")
    # transform list of dicts to dict
    start = time.perf_counter()
    phrog_categories = {
        k: v for x in list_phrog_categories for k, v in x.items()}
    end = time.perf_counter()
    runtime = end - start
    print(f"Done phrog_categories in {runtime:0.8f}")
    # print(phrog_categories)

    # alternative prediction using shared dictionary - not working really
    # TODO: archive functions and remove from versioned script
    # phrog_categories = batch_exec2(phrogs_to_predict, vectors, func_dict_df, top_known_phrogs)

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
        # scores = validation(func_dict_df, phrog_categories)
        scores = parallel_validation(func_dict_df, phrog_categories)
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
