import fasttext_train as ft
import word2vec_train as w2v


# ft.visualisation_pipeline(
#     corpus_path="results/vir2000_numbered.pickle",
#     vector_size=150,
#     window=3,
#     min_count=5,
#     epochs=200,
#     workers=8,
#     lr_start=0.25,
#     visual_path="plots/vir2000_lr02_e20_d150_w3_minc5_nonsorted.html",
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
    lr_min=0.001,
    hs=0,
    negative=20,
    visual_path="plots/vir2000_test1_ns001_lr0005_lrmin0001_v250_w2_e1000_hs0_neg20.html",
    callbacks=[w2v.TrainLogger()]
)