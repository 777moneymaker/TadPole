import fasttext_train as ft
import word2vec_train as w2v
import utils
from pathlib import Path
import bayes_optimization as bay
# import evaluation as eval
from gensim.models import Word2Vec


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
pipe = ft.FastTextPipeline(
   corpus_path="results/fasttext_consensus_corpus.pickle",
   output_prefix="ft_consensus_test",
   metadata="Data/metadata_phrog.pickle",
   vector_size=20,
   window=2,
   min_count=2,
   epochs=10,
   workers=40,
    # lr_start=0.005,
    # lr_min=0.0001,
   lr_start=0.1,
   lr_min=0.0001,
   hs=0,
   negative=75,
   ns_exp=-0.75,
   visualise_model=False,
   encoded=False,
   save_model=False
)

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

# callback_test
# pipe = w2v.Word2VecPipeline(
#     corpus_path="results/virall_noncoded_14-04-2023.pickle",
#     output_prefix="callback_test",
#     metadata="Data/metadata_phrog.pickle",
#     vector_size=40,
#     window=2,
#     min_count=2,
#     epochs=5,
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

# # aus_w2v_2nd_followup_domainreduction
# pipe = w2v.Word2VecPipeline(
#     corpus_path="results/virall_noncoded_14-04-2023.pickle",
#     output_prefix="aus_w2v_2nd_followup_domainreduction",
#     metadata="Data/metadata_phrog.pickle",
#     vector_size=80,
#     window=2,
#     min_count=2,
#     epochs=200,
#     workers=40,
#     lr_start=0.005,
#     lr_min=0.0001,
#     hs=0,
#     negative=50,
#     ns_exp=-0.1,
#     sample=0.001,
#     callbacks=[w2v.TrainLogger()],
#     visualise_model=False,
#     encoded=False,
#     save_model= False
# )

# aus_w2v_bigcorpus
# pipe = w2v.Word2VecPipeline(
#     corpus_path="results/big_corpus_w2v.pickle",
#     output_prefix="aus_w2v_bigcorpus",
#     metadata="Data/metadata_phrog.pickle",
#     vector_size=80,
#     window=2,
#     min_count=2,
#     epochs=200,
#     workers=40,
#     lr_start=0.005,
#     lr_min=0.0001,
#     hs=0,
#     negative=50,
#     ns_exp=-0.1,
#     sample=0.001,
#     callbacks=[w2v.TrainLogger()],
#     visualise_model=False,
#     encoded=False,
#     save_model= False
# )

# aus_w2v_auscorpus
# pipe = w2v.Word2VecPipeline(
#     corpus_path="results/AUS.corpus.pkl",
#     output_prefix="aus_w2v_auscorpus",
#     metadata="Data/metadata_phrog.pickle",
#     vector_size=180,
#     window=2,
#     min_count=2,
#     epochs=200,
#     workers=40,
#     lr_start=0.005,
#     lr_min=0.0001,
#     hs=0,
#     negative=50,
#     ns_exp=-0.1,
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
#     'lr_min': (0.000001, 0.2),
# }

# aus_w2v_2nd_followup_domainreduction
# hypers = {
#     'window': (2, 30),
#     'ns_exp': (0.01, 0.95),
#     'lr_start': (0.0009, 0.1),
#     'lr_min': (0.000009, 0.01),
#     'negative': (30, 120),
#     'min_count': (2, 30)
# }

# aus_w2v_bigcorpus
# hypers = {
#     'epochs': (75, 500),
#     'vector_size': (80, 300),
#     'window': (2, 75),
#     'ns_exp': (-0.95, 0.95),
#     'lr_start': (0.0009, 0.1),
#     'lr_min': (0.000009, 0.01),
#     'negative': (30, 120),
#     'min_count': (2, 30)
# }

# aus_w2v_auscorpus
# hypers = {
#     #'epochs': (75, 500), 200
#     #'vector_size': (80, 300), 180
#     'window': (1, 15),
#     'ns_exp': (-0.95, 0.1),
#     'lr_start': (0.00001, 0.1),
#     'lr_min': (0.0000001, 0.01),
#     'negative': (20, 60),
#     'min_count': (10, 45)
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
# hypers = {
#     'vector_size': (75, 300),
#     # 'epochs': (200, 400), # 250
#     'window': (2, 25),
#     'ns_exp': (-0.95, 0.95),
#     'lr_start': (0.000001, 0.1),
#     'lr_min': (0.000001, 0.2),
# }

# aus_ft
hypers = {
    'vector_size': (50, 250),
    # 'epochs': 500, 
    'ns_exp': (-0.1, 0.7),
    'lr_start': (0.005, 0.25),
    'lr_min': (0.00001, 0.01),
    'negative': (45, 300),
    'max_n': (5, 12), #ngram max
    'min_n': (2, 8) #ngram min
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


# bayes = bay.BayesianOptimizer(pipe, hypers, 10, 35, "aus_w2v_auscorpus", Path("./logs/aus_w2v_auscorpus"), aquisition_function='ucb', kappa=7.2, domain_reduction=False)
bayes = bay.BayesianOptimizer(pipe, hypers, 2, 2, "ft_consensus_test", Path("./logs/ft_consensus_test"), aquisition_function='ucb', kappa=10, domain_reduction=False)
bayes.optimize()

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
# >>> model = Word2Vec.load("logs/aus_w2v_auscorpus/aus_w2v_auscorpus_ns-03451567394644681_lr0024171862536102377_lrmin0006037547436526886_d180_w2_e200_hs0_neg34_mincount34.model")
# >>> embedding = w2v.umap_reduce(model.wv, 3)
# UMAP Magic |████████████████████████████████████████| 1 in 33.1s (0.03/s)
# >>> visual = w2v.model_visualise(model.wv, embedding, "plots/aus_w2v_auscorpus_ns-03451567394644681_lr0024171862536102377_lrmin0006037547436526886_d180_w2_e200_hs0_neg34_mincount34.html", False)
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

# pipe = w2v.Word2VecPipeline(
#     corpus_path="results/virall_noncoded_14-04-2023.pickle",
#     output_prefix="sg_eval_test_rangepower+pyarrow_mincouttest",
#     metadata="Data/metadata_phrog.pickle",
#     vector_size=120,
#     window=4,
#     min_count=10,
#     epochs=50,
#     workers=40,
#     lr_start=0.016186931560335408,
#     lr_min=0.0015449634769682278,
#     hs=0,
#     sg=1,
#     negative=50,
#     ns_exp=0.6768068921197985,
#     sample=0.001,
#     callbacks=[w2v.TrainLogger()],
#     visualise_model=True,
#     encoded=False,
#     save_model= True
# )
# pipe.run()

# pipe = w2v.Word2VecPipeline(
#     corpus_path="results/virall_noncoded_14-04-2023.pickle",
#     output_prefix="aus_w2v_bigsmall_test",
#     metadata="Data/metadata_phrog.pickle",
#     vector_size=189,
#     window=7,
#     min_count=19,
#     epochs=197,
#     workers=40,
#     lr_start=0.038334,
#     lr_min=0.003185,
#     hs=0,
#     negative=51,
#     ns_exp=-0.572082,
#     sample=0.001,
#     callbacks=[w2v.TrainLogger()],
#     visualise_model=False,
#     encoded=False,
#     save_model= True
# )
# pipe.run()


# import word2vec_train as w2v
# import utils
# from pathlib import Path
# import bayes_optimization as bay
# import evaluation
# from gensim.models import Word2Vec
# import numpy as np
# from scipy.linalg import orthogonal_procrustes

# ## merge models
# model_small = Word2Vec.load("logs/aus_w2v_2nd/aus_w2v_2nd_ns06768068921197985_lr0016186931560335408_lrmin00015449634769682278_d120_w4_e500_hs0_neg50_mincount2.model")
# model_big = Word2Vec.load("logs/aus_w2v_bigcorpus/aus_w2v_bigcorpus_ns-035821676466215857_lr004119140981062489_lrmin00015265021199645495_d112_w4_e211_hs0_neg33_mincount22.model")
# # model_small.wv.add_vectors(model_big.wv.index_to_key, model_big.wv.vectors)
# common_vocab = set(model_small.wv.index_to_key).union(set(model_big.wv.index_to_key))
# merged_model = Word2Vec(vector_size=model_small.vector_size, min_count=1)
# merged_model.build_vocab([list(common_vocab)])
# # for word in common_vocab:
# #     merged_model.wv[word] = (model_small.wv[word] + model_big.wv[word]) / 2
# # OR:
# model_small.build_vocab([list(model_big.wv.index_to_key)], update=True)
# model_small.train([list(model_big.wv.index_to_key)], total_examples=model_small.corpus_count, epochs=model_small.epochs)

# model_big.build_vocab([list(model_small.wv.index_to_key)], update=True)
# model_big.train([list(model_small.wv.index_to_key)], total_examples=model_big.corpus_count, epochs=model_big.epochs)


# model_small = Word2Vec.load("logs/aus_w2v_2nd/aus_w2v_2nd_ns06768068921197985_lr0016186931560335408_lrmin00015449634769682278_d120_w4_e500_hs0_neg50_mincount2.model")
# model_big = Word2Vec.load("logs/aus_w2v_bigcorpus/aus_w2v_bigcorpus_ns-035821676466215857_lr004119140981062489_lrmin00015265021199645495_d112_w4_e211_hs0_neg33_mincount22.model")
# model_small.build_vocab([list(model_big.wv.index_to_key)], update=True)
# model_small.train([list(model_big.wv.index_to_key)], total_examples=model_big.corpus_count, epochs=model_big.epochs, callbacks=[w2v.TrainLogger()], compute_loss=True)



# model_small = Word2Vec.load("logs/aus_w2v_2nd/aus_w2v_2nd_ns06768068921197985_lr0016186931560335408_lrmin00015449634769682278_d120_w4_e500_hs0_neg50_mincount2.model")
# model_big = Word2Vec.load("logs/aus_w2v_bigcorpus/aus_w2v_bigcorpus_ns-035821676466215857_lr004119140981062489_lrmin00015265021199645495_d112_w4_e211_hs0_neg33_mincount22.model")
# common_vocab = set(model_small.wv.index_to_key).union(set(model_big.wv.index_to_key))
# merged_model = Word2Vec(vector_size=model_small.vector_size, min_count=1)
# merged_model.build_vocab([list(common_vocab)])
# merged_model.train([list(model_big.wv.index_to_key)], total_examples=model_big.corpus_count, epochs=model_big.epochs, callbacks=[w2v.TrainLogger()], compute_loss=True)

# funcs = utils.read_metadata(Path("Data/metadata_phrog.pickle"))
# model_small = Word2Vec.load("train_test/aus_w2v_2nd_mix_ns06768068921197985_lr0016186931560335408_lrmin00015449634769682278_d112_w4_e500_hs0_neg50_mincount2.model")
# model_big = Word2Vec.load("logs/aus_w2v_bigcorpus/aus_w2v_bigcorpus_ns-035821676466215857_lr004119140981062489_lrmin00015265021199645495_d112_w4_e211_hs0_neg33_mincount22.model")
# # common_vocab = set(model_small.wv.index_to_key).intersection(set(model_big.wv.index_to_key))
# common_vocab = set(model_small.wv.index_to_key).union(set(model_big.wv.index_to_key))
# merged_model = Word2Vec(vector_size=model_small.vector_size, min_count=1)
# merged_model.build_vocab([list(common_vocab)])
# for word in common_vocab:
#     merged_model.wv[word] = (model_small.wv[word] + model_big.wv[word]) / 2
# eval = evaluation.prediction(func_dict=funcs, model=merged_model, model_name="merged_model_avg")


# merged_model = Word2Vec(vector_size=model_small.vector_size + model_big.vector_size, min_count=1)
# merged_model.build_vocab([list(model_small.wv.index_to_key) + list(model_big.wv.index_to_key)])
# for word in merged_model.wv.index_to_key:
#     merged_model.wv[word] = np.concatenate((model_small.wv[word], model_big.wv[word]))


# common_vocab = set(model_small.wv.index_to_key).intersection(set(model_big.wv.index_to_key))
# vecs1 = np.array([model_small.wv[word] for word in common_vocab])
# vecs2 = np.array([model_big.wv[word] for word in common_vocab])
# M, _ = orthogonal_procrustes(vecs1, vecs2)
# vecs2_aligned = vecs2.dot(M)
# merged_model = Word2Vec(vector_size=model_small.vector_size, min_count=1)
# merged_model.build_vocab([list(common_vocab)])
# for i, word in enumerate(common_vocab):
#     merged_model.wv[word] = (vecs1[i] + vecs2_aligned[i]) / 2


# common_vocab = set(model_small.wv.index_to_key).intersection(set(model_big.wv.index_to_key))
# total_counts = {}
# for word in common_vocab:
#     total_counts[word] = model_small.wv.get_vecattr(word, 'count') + model_big.wv.get_vecattr(word, 'count')
# weights1 = {word: model_small.wv.get_vecattr(word, 'count') / total_counts[word] for word in common_vocab}
# weights2 = {word: model_big.wv.get_vecattr(word, 'count') / total_counts[word] for word in common_vocab}
# merged_model = Word2Vec(vector_size=model_small.vector_size, min_count=1)
# merged_model.build_vocab([list(common_vocab)])
# for word in common_vocab:
#     merged_model.wv[word] = weights1[word] * model_small.wv.get_vecattr(word, 'count') + weights2[word] * model_big.wv.get_vecattr(word, 'count')

# for word in common_vocab:
#     count = (model_small.wv.get_vecattr(word, 'count') + model_big.wv.get_vecattr(word, 'count'))
#     merged_vector = (model_small.wv[word] * (model_small.wv.get_vecattr(word, 'count') / count)) + (model_big.wv[word] * (model_big.wv.get_vecattr(word, 'count') / count))
#     merged_model.wv[word] = merged_vector

# pipe = w2v.Word2VecPipeline(
#     corpus_path="results/virall_noncoded_14-04-2023.pickle",
#     output_prefix="aus_w2v_2nd_mix",
#     metadata="Data/metadata_phrog.pickle",
#     vector_size=112,
#     window=4,
#     min_count=2,
#     epochs=500,
#     workers=63,
#     lr_start=0.016186931560335408,
#     lr_min=0.0015449634769682278,
#     hs=0,
#     negative=50,
#     ns_exp=0.6768068921197985,
#     callbacks=[w2v.TrainLogger()],
#     visualise_model=False,
#     encoded=False,
#     save_model= True
# )
# pipe.run()

# train_test/aus_w2v_2nd_mix_ns06768068921197985_lr0016186931560335408_lrmin00015449634769682278_d112_w4_e500_hs0_neg50_mincount2.model


