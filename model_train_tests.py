import fasttext_train as ft


ft.visualisation_pipeline(
    corpus_path="results/vir2000_numbered.pickle",
    vector_size=150,
    window=3,
    min_count=5,
    epochs=200,
    workers=8,
    lr_start=0.2,
    visual_path="plots/vir2000_lr02_e200_d150_w3_minc5_nonsorted.html"
)