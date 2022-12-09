from pathlib import Path
from typing import Union
from gensim.models import FastText, Word2Vec

import utils


def prediction(
    func_dict: dict,
    model: Union[FastText, Word2Vec]
):
    # TODO: replace phrogs with functions
    
    # TODO: select top50 matches with known function
    vectors = model.wv
    result = vectors.most_similar('joker', topn = 5) # placeholder
    #print(result)

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