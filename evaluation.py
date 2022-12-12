from pathlib import Path
import pandas as pd
from typing import Union
from gensim.models import FastText, Word2Vec
from collections import defaultdict

import utils

def sum_tuples(lst: tuple[str, float]):
    d = defaultdict(float)
    for category, prob in lst:
        d[category] += prob
    
    return list(d.items())
    
def mean_tuples(lst: tuple[str, float]):
    d = defaultdict(float)
    occurs = defaultdict(int)
    for category, prob in lst:
        d[category] += prob
        occurs[category] += 1
    for category in d:
        d[category] /= occurs[category]
    return list(d.items())

def prediction(
    func_dict: dict,
    model: Union[FastText, Word2Vec]
):
    # convert dict to pandas dataframe or read it directly
    df = pd.DataFrame(func_dict.items(), columns=['phrog_id', 'category'])
    #df = pd.read_table('Data/metadata-phrog.tsv', header=0)

    # create a list of phrogs with known function
    known_func_phrog_list = df[(df['category'] != 'unknown function')]['phrog_id'].tolist()
    # TODO: Consider making this a set, O(1) membership checking
    # known_func_phrog_list = set(known_func_phrog_list)

    # select matches with known function
    #for phrog in known_func_phrog_list:   # <-- TODO: consider changing it to multiprocessing

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

    for phrog in known_func_phrog_list[:10]: # placeholder, model wasn't trained for all phrogs yet, tested to work on phrogs1-10
        vectors = model.wv  
        result = vectors.most_similar(phrog, topn = 40_000) # Use _ for clarity # <-- TODO: looking for ideas how to optimize this
        #result = (list(filter(lambda x: 'joker' not in x[0], result)))  # to remove jokers from result; turns out mergeddf_to_tuple isnt returning them anyway so far

        # replace phrogs with functions
        tuples_to_df = pd.DataFrame(result, columns=['phrog_id', 'probability'])
        merged = tuples_to_df.merge(df, on='phrog_id') # Same col, just use on=
        
        # just use loc, add .head immeadiately
        merged = merged.loc[merged['category'] != 'unknown function'].head(50)
        merged_id_category = merged[["category", "probability"]]

        # Begin scoring
        tpls = list(merged_id_category.apply(tuple, 1))
        key_func = lambda x: x[1]
        
        # 3 scoring functions
        # mx: max value for a category
        # sum: max value after summing prob for each category
        # mean: max value after taking a mean prob for each category
        mx = max(tpls, key=key_func)
        summed = max(sum_tuples(tpls), key=key_func)
        mean = max(mean_tuples(tpls), key=key_func)

        
        phrog_categories[phrog] = {
            "max": mx,
            "sum": summed,
            "mean": mean
        }
        # End scoring

    # TODO: validation


#
######## --------------------------------- ########
# Temporary code for running as a separate script #  VVV

# read phrog:function dictionary
func = utils.read_metadata(Path("Data/metadata_phrog.dill")) 
# read trained model
mod =  Word2Vec.load('train_test/test.model')
#mod = FastText.load() 

# run
prediction(func, mod)
