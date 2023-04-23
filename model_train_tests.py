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
# aus_w2v_2nd
# pipe = w2v.Word2VecPipeline(
#     corpus_path="results/virall_noncoded_14-04-2023.pickle",
#     output_prefix="aus_w2v_2nd",
#     metadata="Data/metadata_phrog.pickle",
#     vector_size=120,
#     window=2,
#     min_count=2,
#     epochs=500,
#     workers=40,
#     lr_start=0.005,
#     lr_min=0.0001,
#     hs=0,
#     negative=50,
#     ns_exp=-0.1,
#     callbacks=[w2v.TrainLogger()],
#     visualise_model=False,
#     encoded=False,
#     save_model= False
# )

## aus_ft
# pipe = ft.FastTextPipeline(
#    corpus_path="results/virall_encode_02-04-2023.pickle",
#    output_prefix="aus_ft",
#    metadata="Data/metadata_02-04-2023.pickle",
#    vector_size=20,
#    window=2,
#    min_count=2,
#    epochs=450,
#    workers=40,
#     # lr_start=0.005,
#     # lr_min=0.0001,
#    lr_start=0.1,
#    lr_min=0.0001,
#    hs=0,
#    negative=75,
#    ns_exp=-0.75,
#    visualise_model=False,
#    encoded=True,
#    save_model=False
# )

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

# aus_w2v_sg
# pipe = w2v.Word2VecPipeline(
#     corpus_path="results/virall_noncoded_14-04-2023.pickle",
#     output_prefix="aus_w2v_sg",
#     metadata="Data/metadata_phrog.pickle",
#     vector_size=120,
#     window=2,
#     min_count=2,
#     epochs=500,
#     workers=40,
#     lr_start=0.005,
#     lr_min=0.0001,
#     hs=0,
#     sg=1,
#     negative=50,
#     ns_exp=-0.1,
#     callbacks=[w2v.TrainLogger()],
#     visualise_model=False,
#     encoded=False,
#     save_model= False
# )

# aus_w2v_sg_2nd
# pipe = w2v.Word2VecPipeline(
#     corpus_path="results/virall_noncoded_14-04-2023.pickle",
#     output_prefix="aus_w2v_sg_2nd",
#     metadata="Data/metadata_phrog.pickle",
#     vector_size=150,
#     window=2,
#     min_count=2,
#     epochs=450,
#     workers=40,
#     lr_start=0.005,
#     lr_min=0.0001,
#     hs=0,
#     sg=1,
#     negative=50,
#     ns_exp=-0.1,
#     callbacks=[w2v.TrainLogger()],
#     visualise_model=False,
#     encoded=False,
#     save_model= False
# )

# aus_w2v_cbow_sampleword
# pipe = w2v.Word2VecPipeline(
#     corpus_path="results/virall_noncoded_14-04-2023.pickle",
#     output_prefix="aus_w2v_cbow_sampleword",
#     metadata="Data/metadata_phrog.pickle",
#     vector_size=120,
#     window=2,
#     min_count=2,
#     epochs=450,
#     workers=40,
#     lr_start=0.005,
#     lr_min=0.0001,
#     hs=0,
#     sg=0,
#     negative=50,
#     ns_exp=-0.1,
#     sample=0.001,
#     callbacks=[w2v.TrainLogger()],
#     visualise_model=False,
#     encoded=False,
#     save_model= False
# )

# aus_w2v_sghs
# pipe = w2v.Word2VecPipeline(
#     corpus_path="results/virall_noncoded_14-04-2023.pickle",
#     output_prefix="aus_w2v_sghs",
#     metadata="Data/metadata_phrog.pickle",
#     vector_size=120,
#     window=15,
#     min_count=2,
#     epochs=250,
#     workers=40,
#     lr_start=0.03324065123940248,
#     lr_min=0.0006380197844771706,
#     hs=1,
#     sg=1,
#     negative=0,
#     ns_exp=0.5515559349266264,
#     sample=0.001,
#     callbacks=[w2v.TrainLogger()],
#     visualise_model=False,
#     encoded=False,
#     save_model= False
# )

# hypers = {
#     'vector_size': (50, 200),
#     'epochs': (100, 500),
#     'ns_exp': (-0.75, 0.75),
#     'lr_start': (0.001, 0.5),
#     'lr_min': (0.0001, 0.1),
#     'negative': (45, 135)
# }

# quick_hypers = {
#     'ns_exp': (-0.9, 0.9),
#     'negative': (45, 100),
# }

# aus_w2v_2nd
# hypers = {
#     'window': (2, 5),
#     'ns_exp': (0.01, 0.95),
#     'lr_start': (0.000001, 0.1),
#     'lr_min': (0.000001, 0.2)
# }

# aus_w2v_sg
# hypers = {
#     'vector_size': (100, 300),
#     'epochs': (400, 750),
#     'window': (2, 5),
#     'ns_exp': (0.01, 0.95),
#     'lr_start': (0.000001, 0.1),
#     'lr_min': (0.000001, 0.2),
#     'negative': (0, 135)
# }

# # aus_w2v_sg_2nd
# hypers = {
#     # 'vector_size': (100, 300), 150
#     # 'epochs': (200, 550), 450
#     'window': (2, 25),
#     'ns_exp': (0.01, 0.95),
#     'lr_start': (0.0009, 0.1),
#     'lr_min': (0.000009, 0.01),
#     'negative': (0, 100),
#     'sample': (0, 0.00001)
# }

# aus_w2v_cbow_sampleword
# hypers = {
#     'window': (5, 25),
#     'ns_exp': (0.01, 0.95),
#     'lr_start': (0.000001, 0.1),
#     'lr_min': (0.000001, 0.2),
#     'sample': (0, 0.0009)
# }

# aus_w2v_sghs
hypers = {
    'vector_size': (75, 300),
    # 'epochs': (200, 400), # 250
    'window': (2, 25),
    'ns_exp': (-0.95, 0.95),
    'lr_start': (0.000001, 0.1),
    'lr_min': (0.000001, 0.2),
}

# aus_ft
# hypers = {
#     'vector_size': (50, 250),
#     # 'epochs': 500, 
#     'ns_exp': (-0.1, 0.7),
#     'lr_start': (0.005, 0.25),
#     'lr_min': (0.00001, 0.01),
#     'negative': (45, 300),
#     'max_n': (5, 12), #ngram max
#     'min_n': (2, 8) #ngram min
# }

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

# bayes = bay.BayesianOptimizer(pipe, hypers, 12, 23, "aus_w2v_sghs", Path("./logs/aus_w2v_sghs"), aquisition_function='ucb', kappa=10)
# bayes.optimize()

# aus_w2v_sg even categories 59% model word tweak - veeery slow
# pipe = w2v.Word2VecPipeline(
#     corpus_path="results/virall_noncoded_14-04-2023.pickle",
#     output_prefix="aus_w2v_sg_even_categories_59_wordstweak",
#     metadata="Data/metadata_phrog.pickle",
#     vector_size=155,
#     window=25,
#     min_count=2,
#     epochs=500,
#     workers=40,
#     lr_start=0.018327240440219397,
#     lr_min=0.06047517970145826,
#     hs=0,
#     sg=1,
#     negative=82,
#     ns_exp=0.38757200932613456,
#     callbacks=[w2v.TrainLogger()],
#     visualise_model=True,
#     encoded=False,
#     save_model= True
# )
# pipe.run()

# >>> import fasttext_train as ft
# >>> from gensim.models import FastText
# >>> model = FastText.load("logs/aus_ft/aus_ft_ns0032283357693546205_lr003868130251582819_lrmin0009034985133725956_d235_w2_e450_hs0_neg146_maxn5_minn6.model")
# >>> embedding = ft.umap_reduce(model.wv, 3)
# UMAP Magic |████████████████████████████████████████| 1 in 45.1s (0.02/s)
# >>> visual = ft.model_visualise(model.wv, embedding, "plots/aus_ft_ns0032283357693546205_lr003868130251582819_lrmin0009034985133725956_d235_w2_e450_hs0_neg146_maxn5_minn6.html", True)

# >>> import word2vec_train as w2v
# >>> from gensim.models import Word2Vec
# >>> model = Word2Vec.load("logs/aus_w2v_sghs/aus_w2v_sghs_ns-095_lr01_lrmin1e-06_d89_w2_e250_hs1_neg0_mincount2.model")
# >>> embedding = w2v.umap_reduce(model.wv, 3)
# UMAP Magic |████████████████████████████████████████| 1 in 33.1s (0.03/s)
# >>> visual = w2v.model_visualise(model.wv, embedding, "plots/aus_w2v_sghs_ns-095_lr01_lrmin1e-06_d89_w2_e250_hs1_neg0_mincount2.html", False)
# on 0:     INFO >>> Loading dill with phrog metadata
# Gathering phrog metadata and embedding data |████████████████████████████████████████| 1 in 0.1s (9.19/s)
# Generating visualisation |████████████████████████████████████████| 1 in 1.3s (0.75/s)


# earliest test of sg to compare to best aus_w2v_2nd
# import word2vec_train as w2v
# from gensim.models import Word2Vec
# pipe = w2v.Word2VecPipeline(corpus_path="results/virall_noncoded_14-04-2023.pickle", 
#                         output_prefix="sg_w2v_sampletest_lowsample_wordy",
#                         metadata="Data/metadata_phrog.pickle",
#                         vector_size=120,
#                         window=20,
#                         min_count=2,
#                         epochs=500,
#                         workers=40,
#                         lr_start=0.016186931560335408,
#                         lr_min=0.0015449634769682278,
#                         hs=0,
#                         sg=1,
#                         negative=50,
#                         ns_exp=0.6768068921197985,
#                         sample=0.00001,
#                         callbacks=[w2v.TrainLogger()],
#                         visualise_model=True,
#                         encoded=False,
#                         save_model= True)
# pipe.run()

# pipe = w2v.Word2VecPipeline(
#     corpus_path="results/virall_noncoded_14-04-2023.pickle",
#     output_prefix="aus_w2v_cbowns_test_funky",
#     metadata="Data/metadata_phrog.pickle",
#     vector_size=89,
#     window=50,
#     min_count=2,
#     epochs=60,
#     workers=40,
#     lr_start=0.0015449634769682278,
#     lr_min=0.016186931560335408,
#     hs=0,
#     sg=0,
#     negative=100,
#     ns_exp=0.6768068921197985,
#     sample=0.001,
#     callbacks=[w2v.TrainLogger()],
#     visualise_model=True,
#     encoded=False,
#     save_model= True
# )
# pipe.run()

pipe = w2v.Word2VecPipeline(
    corpus_path="results/virall_noncoded_14-04-2023.pickle",
    output_prefix="eval_test_rangepower+pyarrow_mincouttest",
    metadata="Data/metadata_phrog.pickle",
    vector_size=120,
    window=4,
    min_count=25,
    epochs=500,
    workers=40,
    lr_start=0.016186931560335408,
    lr_min=0.0015449634769682278,
    hs=0,
    sg=0,
    negative=50,
    ns_exp=0.6768068921197985,
    sample=0.001,
    callbacks=[w2v.TrainLogger()],
    visualise_model=True,
    encoded=False,
    save_model= True
)
pipe.run()