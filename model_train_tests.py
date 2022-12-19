import fasttext_train as ft
import word2vec_train as w2v
import utils
from pathlib import Path
import evaluation as eval


ft.visualisation_pipeline(
    corpus_path="results/vir2000_numbered.pickle",
    vector_size=150,
    window=2,
    min_count=5,
    epochs=750,
    workers=8,
    lr_start=1,
    lr_min=0.0001,
    negative=75,
    ns_exp=-0.75,
    visual_path="plots/vir2000_lr1_lrmin00001_e750_d150_w2_minc5_neg100_nsexp-075_nonsorted.html",
    # callbacks=[ft.TrainLogger()]
    callbacks=[ft.TrainLogger()]
)

# w2v.visualisation_pipeline(
#     corpus_path="results/vir2000_numbered.pickle",
#     vector_size=250,
#     window=2,
#     min_count=5,
#     epochs=1000,
#     workers=8,
#     lr_start=0.005,
#     lr_min=0.0001,
#     hs=0,
#     negative=75,
#     ns_exp=0.01,
#     visual_path="plots/vir2000_test1_ns001_lr0005_lrmin000001_v250_w2_e1000_hs0_neg75_version2.html",
#     callbacks=[w2v.TrainLogger()]
# )

# func = utils.read_metadata(Path("Data/metadata_phrog.pickle"))
# model = w2v.wv.Word2Vec.load('train_test/test_w2v.model')
# # model = w2v.model_train(
# #     corpus_path="results/vir2000_numbered.pickle",
# #     vector_size=250,
# #     window=2,
# #     min_count=5,
# #     epochs=1000,
# #     workers=8,
# #     lr_start=0.005,
# #     lr_min=0.0001,
# #     hs=0,
# #     negative=75,
# #     ns_exp=-0.1,
# #     callbacks=[w2v.TrainLogger()]
# # )
# # model.save("train_test/vir2000_test1_ns001_lr0005_lrmin000001_v250_w2_e1000_hs0_neg75")
# eval.prediction(func, model, top_known_phrogs=1)