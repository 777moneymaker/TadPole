import fasttext_train as ft
import word2vec_train as w2v


# ft.visualisation_pipeline(
#     corpus_path="results/vir2000_numbered.pickle",
#     vector_size=150,
#     window=2,
#     min_count=5,
#     epochs=750,
#     workers=8,
#     lr_start=0.005,
#     lr_min=0.0001,
#     negative=75,
#     ns_exp=-0.75,
#     visual_path="plots/vir2000_lr0005_lrmin00001_e750_d150_w2_minc5_neg100_nsexp-075_nonsorted.html",
#     # callbacks=[ft.TrainLogger()]
#     callbacks=[ft.TrainLogger()]
# )

w2v.visualisation_pipeline(
    corpus_path="results/vir2000_numbered.pickle",
    vector_size=250,
    window=2,
    min_count=5,
    epochs=1000,
    workers=8,
    lr_start=0.005,
    lr_min=0.0001,
    hs=0,
    negative=75,
    ns_exp=-0.1,
    visual_path="plots/vir2000_test1_ns-01_lr0005_lrmin000001_v250_w2_e1000_hs0_neg75.html",
    callbacks=[w2v.TrainLogger()]
)