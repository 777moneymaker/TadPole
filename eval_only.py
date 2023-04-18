import evaluation
import utils
from gensim.models import Word2Vec
from pathlib import Path


funcs = utils.read_metadata(Path("Data/metadata_phrog.pickle"))
model = Word2Vec.load("train_test/sg_w2v_ns06768068921197985_lr0016186931560335408_lrmin00015449634769682278_d120_w4_e500_hs0_neg50_mincount2.model")
model_name = "sg_w2v_ns06768068921197985_lr0016186931560335408_lrmin00015449634769682278_d120_w4_e500_hs0_neg50_mincount2"
eval = evaluation.prediction(func_dict=funcs, model=model, model_name=model_name)