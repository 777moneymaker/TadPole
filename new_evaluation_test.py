import evaluation_mamciej
import utils
from gensim.models import FastText
from pathlib import Path


funcs = utils.read_metadata(Path("Data/metadata_02-04-2023.pickle"))
model = FastText.load("logs/w2v_optProto/virall_optProto_w2v_ns-049525437065314665_lr004343706147351918_lrmin008947120568403436_d181_w2_e450_hs0_neg48_mincount5.model")
model_name = "virall_optProto_w2v_ns-049525437065314665_lr004343706147351918_lrmin008947120568403436_d181_w2_e450_hs0_neg48_mincount5"
eval = evaluation_mamciej.prediction(func_dict=funcs, model=model, model_name=model_name)