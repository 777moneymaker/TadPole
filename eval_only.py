import evaluation
import utils
from gensim.models import Word2Vec
from pathlib import Path


funcs = utils.read_metadata(Path("Data/metadata_phrog.pickle"))
model = Word2Vec.load("logs/aus_w2v_auscorpus/aus_w2v_auscorpus_ns-03451567394644681_lr0024171862536102377_lrmin0006037547436526886_d180_w2_e200_hs0_neg34_mincount34.model")
model_name = "aus_w2v_auscorpus"
eval = evaluation.prediction(func_dict=funcs, model=model, model_name="aus_w2v_auscorpus", raw_out=True, evaluate_mode=False)