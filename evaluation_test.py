import evaluation
import utils
from gensim.models import FastText
from pathlib import Path


funcs = utils.read_metadata(Path("Data/metadata_02-04-2023.pickle"))
model = FastText.load("logs/ft_optProto/ft_virall_optProto_ns024751227405245746_lr0_lrmin00074157752401361075_d62_w2_e507_hs0_neg137_maxn8_minn2.model")
model_name = "ft_virall_optProto_ns024751227405245746_lr0_lrmin00074157752401361075_d62_w2_e507_hs0_neg137_maxn8_minn2"
eval = evaluation.prediction(func_dict=funcs, model=model, model_name=model_name)