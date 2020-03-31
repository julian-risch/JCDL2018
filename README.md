# JCDL2018

## Citation

If you use our work, please cite our paper [**My Approach = Your Apparatus? Entropy-Based Topic Modeling on Multiple Domain-Specific Text Collections**](https://github.com/julian-risch/JCDL2018/raw/master/risch2018approach.pdf) as follows:

    @inproceedings{risch2018approach,
    author = {Risch, Julian and Krestel, Ralf},
    booktitle = {Proceedings of the Joint Conference on Digital Libraries (JCDL)},
    pages = {283-292},
    title = {My Approach = Your Apparatus? Entropy-Based Topic Modeling on Multiple Domain-Specific Text Collections},
    year = {2018}
    }

## Implementation
* `TopicModelCcLDA.java` implements entity-based cross-collection latent Dirichlet allocation
* `RunTopicModel.java` starts the training of the topic model and the following evalution
* `CorpusBlogPostsFromFile.java` loads a corpus from a file.
* `CorpusToy.java` loads a small example corpus of a few documents defined in the code.
