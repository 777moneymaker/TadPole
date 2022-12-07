import fasttext_train as ft


ft.fasttext_pipeline(
    corpus_path="results/vir2000_numbered.pickle",
    numbered=True,
    vector_size=100,
    window=5,
    min_count=1,
    epochs=10,
    workers=4,
    max_n=4,
    min_n=4,
    visual_path="plots/vir2000.html"
)