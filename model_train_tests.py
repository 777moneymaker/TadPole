import fasttext_train as ft
import word2vec_train as w2v
import utils
from pathlib import Path
import bayes_optimization as bay
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

# w2v.evaluation_pipeline(
#     corpus_path="results/virall_encode_better.pickle",
#     output_prefix="virall_coded_17-03",
#     vector_size=150,
#     window=2,
#     min_count=5,
#     epochs=500,
#     workers=40,
#     lr_start=0.005,
#     lr_min=0.0001,
#     hs=0,
#     negative=75,
#     ns_exp=-0.1,
#     callbacks=[w2v.TrainLogger()],
#     visualise_model=False,
#     encoded=True
# )

#ft.evaluation_pipeline_exec(
#    corpus_path="results/virall_encode_better.pickle",
#    output_prefix="virall_encode_coded_17-03_ft",
#    vector_size=150,
#    window=2,
#    min_count=5,
#    epochs=500,
#    workers=40,
#    lr_start=0.005,
#    lr_min=0.0001,
#    negative=75,
#    ns_exp=-0.75,
#    visualise_model=False,
#    encoded=True
#    # callbacks=[ft.TrainLogger()]
#)

########### OOP version

# w2v
# pipe = w2v.Word2VecPipeline(
#     corpus_path="results/virall_encode_02-04-2023.pickle",
#     output_prefix="virall_coded_02-04",
#     metadata="Data/metadata_02-04-2023.pickle",
#     vector_size=150,
#     window=2,
#     min_count=5,
#     epochs=500,
#     workers=40,
#     lr_start=0.005,
#     lr_min=0.0001,
#     hs=0,
#     negative=75,
#     ns_exp=-0.1,
#     callbacks=[w2v.TrainLogger()],
#     visualise_model=False,
#     encoded=True,
#     save_model= True
# )
# pipe.run()

# fasttext
# pipe = ft.FastTextPipeline(
#     corpus_path="results/virall_encode_02-04-2023.pickle",
#     output_prefix="ft_virall_coded_02-04",
#     metadata="Data/metadata_02-04-2023.pickle",
#     vector_size=150,
#     window=2,
#     min_count=5,
#     epochs=500,
#     workers=40,
#     lr_start=0.005,
#     lr_min=0.0001,
#     negative=75,
#     ns_exp=-0.75,
#     visualise_model=False,
#     encoded=True,
# )
# pipe.run()

# opt
pipe = w2v.Word2VecPipeline(
    corpus_path="results/virall_noncoded_14-04-2023.pickle",
    output_prefix="logging_test_w2v_noncoded",
    metadata="Data/metadata-phrog.pickle",
    vector_size=20,
    window=2,
    min_count=5,
    epochs=5,
    workers=40,
    lr_start=0.005,
    lr_min=0.0001,
    hs=0,
    negative=75,
    ns_exp=-0.1,
    callbacks=[w2v.TrainLogger()],
    visualise_model=False,
    encoded=True,
    save_model= False
)

#pipe = ft.FastTextPipeline(
#    corpus_path="results/virall_encode_02-04-2023.pickle",
#    output_prefix="ft_virall_evalSpeedup",
#    metadata="Data/metadata_02-04-2023.pickle",
#    vector_size=20,
#    window=2,
#    min_count=5,
#    epochs=5,
#    workers=40,
    # lr_start=0.005,
    # lr_min=0.0001,
#    lr_start=0.1,
#    lr_min=0.0001,
#    hs=0,
#    negative=75,
#    ns_exp=-0.75,
#    visualise_model=False,
#    encoded=True,
#    save_model=False
#)

# validate_pipe = ft.FastTextPipeline(
#     corpus_path="results/virall_encode_02-04-2023.pickle",
#     output_prefix="ft_virall_optProto_90Validation",
#     metadata="Data/metadata_02-04-2023.pickle",
#     vector_size=62,
#     window=2,
#     min_count=5,
#     epochs=507,
#     workers=40,
#     max_n=8,
#     min_n=2,
#     lr_start=0.08727363947530192,
#     lr_min=0.0074157752401361075,
#     hs=0,
#     negative=137,
#     ns_exp=0.24751227405245746,
#     visualise_model=True,
#     encoded=True,
#     save_model=True
# )
# validate_pipe.run()

# hypers = {
#     'vector_size': (50, 200),
#     'epochs': (100, 500),
#     'ns_exp': (-0.75, 0.75),
#     'lr_start': (0.001, 0.5),
#     'lr_min': (0.0001, 0.1),
#     'negative': (45, 135)
# }

quick_hypers = {
    'ns_exp': (-0.9, 0.9),
    'negative': (45, 100),
}

# hypers = {
#     'vector_size': (50, 300),
#     'epochs': (100, 700),
#     'ns_exp': (-0.9, 0.9),
#     'lr_start': (0.0001, 0.1),
#     'lr_min': (0.00001, 0.01),
#     'negative': (45, 300),
#     'max_n': (2, 12),
#     'min_n': (2, 12)
# }

# hypers = {
#     'vector_size': (200, 500),
#     'epochs': (750, 1250),
#     'ns_exp': (-0.95, 0.95),
#     'lr_start': (0.0001, 0.5),
#     'lr_min': (0.00001, 0.1),
#     'negative': (0, 300),
# }

bayes = bay.BayesianOptimizer(pipe, quick_hypers, 2, 2, "logging_test_w2v_noncoded", Path("./logs/logging_test_w2v_noncoded"), aquisition_function='ucb', kappa=5)
bayes.optimize()

# >>> import fasttext_train as ft
# >>> from gensim.models import FastText
# >>> model = FastText.load("logs/ft_optProto/ft_virall_optProto_ns024751227405245746_lr0_lrmin00074157752401361075_d62_w2_e507_hs0_neg137_maxn8_minn2.model")
# >>> embedding = ft.umap_reduce(model.wv, 3)
# UMAP Magic |████████████████████████████████████████| 1 in 45.1s (0.02/s)
# >>> visual = ft.model_visualise(model.wv, embedding, "plots/ft_virall_optProto_ns024751227405245746_lr0_lrmin00074157752401361075_d62_w2_e507_hs0_neg137_maxn8_minn2.html", True)