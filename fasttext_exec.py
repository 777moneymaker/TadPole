import fasttext_train as ft
import argparse
import logging
from alive_progress import alive_bar
from alive_progress.animations.spinners import bouncing_spinner_factory
import utils
from pathlib import Path
from gensim.models import FastText
import gensim.models.fasttext


parser = argparse.ArgumentParser()
parser.add_argument("-c", help="corpus", type=str)
parser.add_argument("-v", help="corpus", type=int)
parser.add_argument("-w", help="corpus", type=int)
parser.add_argument("-m", help="corpus", type=int)
parser.add_argument("-e", help="corpus", type=int)
parser.add_argument("-t", help="corpus", type=int)
parser.add_argument("--lr", help="corpus", type=float)
parser.add_argument("--lr_min", help="corpus", type=float)
parser.add_argument("--max_n", help="corpus", type=int)
parser.add_argument("--min_n", help="corpus", type=int)
parser.add_argument("--sg", help="corpus", type=int)
parser.add_argument("--hs", help="corpus", type=int)
parser.add_argument("--sorted_vocab", help="corpus", type=int)
parser.add_argument("--neg", help="corpus", type=int)
parser.add_argument("--ns_exp", help="corpus", type=float)
parser.add_argument("--debug", help="corpus", type=bool)
parser.add_argument("--model_name", type=str)
args = parser.parse_args()

ft.PHROG_SPINNER
logging.basicConfig(level=logging.DEBUG)
with alive_bar(title = "Loading corpus",  dual_line = True, spinner = ft.PHROG_SPINNER) as bar:
    sentences = utils.read_corpus(Path(args.c))
    bar()

with alive_bar(title = "Creating model",  dual_line = True, spinner = ft.PHROG_SPINNER) as bar:
    model = FastText(
        vector_size=args.v,
        window=args.w,
        min_count=args.m,
        # sentences=sentences,
        epochs=args.e,
        workers=args.t,
        alpha=args.lr,
        min_alpha=args.lr_min,
        max_n=args.max_n,
        min_n=args.min_n,
        sg=args.sg,
        hs=args.hs,
        negative=args.neg,
        ns_exponent=args.ns_exp,
        sorted_vocab=args.sorted_vocab)
        # model = FastText()
        # print(model.__dict__)
    model.build_vocab(sentences)
        # print(model.corpus_count)
        # print(model.epochs)
    model.train(corpus_iterable=sentences, 
        total_examples=model.corpus_count,
        total_words=model.corpus_total_words, 
        epochs=model.epochs,
        # start_alpha=lr_start,
        # end_alpha=lr_min,
        compute_loss=True,
        # report_delay=0.5,
        callbacks=[ft.TrainLogger()])
    # model.lifecycle_events
    model_path = f"train_test/{args.model_name}.model"
    model.save(model_path)
    bar()

