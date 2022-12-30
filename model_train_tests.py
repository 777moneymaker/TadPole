import fasttext_train as ft
import word2vec_train as w2v
import utils
from pathlib import Path
# import evaluation as eval


# ft.visualisation_pipeline(
#     corpus_path="results/vir2000_numbered.pickle",
#     vector_size=150,
#     window=2,
#     min_count=5,
#     epochs=750,
#     workers=8,
#     lr_start=1,
#     lr_min=0.0001,
#     negative=75,
#     ns_exp=-0.75,
#     visual_path="plots/vir2000_lr1_lrmin00001_e750_d150_w2_minc5_neg100_nsexp-075_nonsorted.html",
#     # callbacks=[ft.TrainLogger()]
#     callbacks=[ft.TrainLogger()]
# )

# ft.visualisation_pipeline_exec(
#     corpus_path="results/vir2000_numbered.pickle",
#     vector_size=150,
#     window=2,
#     min_count=5,
#     epochs=5,
#     workers=8,
#     lr_start=0.005,
#     lr_min=0.0001,
#     negative=75,
#     ns_exp=-0.75,
#     visual_path="plots/vir2000_lr1_lrmin00001_e5_d150_w2_minc5_neg100_nsexp-075_nonsorted.html",
#     # callbacks=[ft.TrainLogger()]
# )

# w2v.visualisation_pipeline(
#     corpus_path="results/vir2000_numbered.pickle",
#     vector_size=25,
#     window=2,
#     min_count=5,
#     epochs=15,
#     workers=12,
#     lr_start=0.005,
#     lr_min=0.0001,
#     hs=0,
#     negative=75,
#     ns_exp=-0.1,
#     visual_path="plots/virall_ns001_lr0005_lrmin00001_v25_w2_e15_hs0_neg75_eventtest.html",
#     callbacks=[w2v.TrainLogger()]
# )

# no idea why, but running model_train here alone is 2x slower, than running whole visualisation_pipeline

# func = utils.read_metadata(Path("Data/metadata_phrog.pickle"))
# model = w2v.wv.Word2Vec.load('train_test/test_w2v.model')
# model = w2v.model_train(
#     corpus_path="results/virall_numbered_noprops.pickle",
#     vector_size=25,
#     window=2,
#     min_count=5,
#     epochs=150,
#     workers=12,
#     lr_start=0.005,
#     lr_min=0.0001,
#     hs=0,
#     negative=75,
#     ns_exp=-0.1,
#     callbacks=[w2v.TrainLogger()]
# )
# model.save("train_test/manual_virall_ns001_lr0005_lrmin00001_v25_w2_e150_hs0_neg75")
# eval.prediction(func, model, top_known_phrogs=1)

# w2v.evaluation_pipeline(
#     corpus_path="results/virall_encode_better.pickle",
#     output_prefix="virall_encode_better_CODED_31-12_DIAG",
#     vector_size=25,
#     window=2,
#     min_count=5,
#     epochs=150,
#     workers=12,
#     lr_start=0.005,
#     lr_min=0.0001,
#     hs=0,
#     negative=75,
#     ns_exp=-0.1,
#     callbacks=[w2v.TrainLogger()],
#     visualise_model=True,
#     encoded=True
# )

ft.evaluation_pipeline_exec(
    corpus_path="results/virall_encode_better.pickle",
    output_prefix="virall_encode_better_CODED_31-12_ft",
    vector_size=25,
    window=2,
    min_count=5,
    epochs=150,
    workers=8,
    lr_start=0.005,
    lr_min=0.0001,
    negative=75,
    ns_exp=-0.75,
    visualise_model=True,
    encoded=True
    # callbacks=[ft.TrainLogger()]
)