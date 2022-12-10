from pathlib import Path
import pandas as pd
from typing import Union
from gensim.models import FastText, Word2Vec

import utils


def prediction(
    func_dict: dict,
    model: Union[FastText, Word2Vec]
):
    # convert dict to pandas dataframe or read it directly
    df = pd.DataFrame(func_dict.items(), columns=['phrog_id', 'category'])
    #df = pd.read_table('Data/metadata-phrog.tsv', header=0)

    # create a list of phrogs with known function
    known_func_phrog_list = df[(df['category'] != 'unknown function')]['phrog_id'].tolist()

    # select matches with known function
    #for phrog in known_func_phrog_list:   # <-- TODO: consider changing it to multiprocessing
    for phrog in known_func_phrog_list[:10]: # placeholder, model wasn't trained for all phrogs yet, tested to work on phrogs1-10
        vectors = model.wv
        result = vectors.most_similar(phrog, topn = 40000) # <-- TODO: looking for ideas how to optimize this
        #result = (list(filter(lambda x: 'joker' not in x[0], result)))  # to remove jokers from result; turns out mergeddf_to_tuple isnt returning them anyway so far
        #print(result)

        # replace phrogs with functions
        tuples_to_df = pd.DataFrame(result, columns=['phrog_id', 'probability'])
        merged = tuples_to_df.merge(df, left_on='phrog_id', right_on='phrog_id')
        merged.drop(merged.index[merged['category'] == 'unknown function'], inplace = True) # remove phrogs with unknown function
        merged = merged.head(50) # select only top 50
        merged_id_category = merged[["category", "probability"]]
        mergeddf_to_tuple = list(merged_id_category.itertuples(index=False, name=None))
        #print(mergeddf_to_tuple)

        # TODO: scoring functions

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
