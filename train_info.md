# Model training progress

## 08.12. 2022
### Questions and issues
- Results of the quick tests
    - Jokers highly clustered
    -  How to code the jokers
    - word2vec is worse already?
    - umap takes its time
- Next tests
    - how to provide data for scoring group
    - random paramter picking with intuition?
- How optimisation will be performed?

### Discussion
> void for now

## 01 - 07.12.2022
### Done
- created pipelines for word2vec and fasttext
    - reading corpus from binary
    - training model
    - dimensionality reduction with UMAP to 3 dimensions
    - gathering data from phrog metadata (functions of phrogs, specifically)
    - visualisation on 3D scatter plot (plotly)
- fasttext pipeline is in own .py file and can be used as module
    > warning: not final form, just the pipeline function is contained as a callable function
### TODO
- add word2vec pipeline as a callable function
- unit tests (probably to be done before the hyperparameter optimisation)
- refactor and design a better flow and module (very dependant on joker encoding and model scoring)
- organise catalogue structure (mostly sync with main - to be done in the future with Miłosz)