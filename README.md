# JCDL2018
Here, we publish source code of our entropy-based cross-collection topic model. This code is a reference implementation of our paper our paper [**My Approach = Your Apparatus? Entropy-Based Topic Modeling on Multiple Domain-Specific Text Collections**](https://hpi.de/fileadmin/user_upload/fachgebiete/naumann/people/risch/risch2018entropy.pdf) at the ACM/IEEE Joint Conference on Digital Libraries 2018.

## Citation

If you use our work, please cite our paper as follows:

    @inproceedings{risch2018approach,
    author = {Risch, Julian and Krestel, Ralf},
    booktitle = {Proceedings of the Joint Conference on Digital Libraries (JCDL)},
    pages = {283-292},
    title = {My Approach = Your Apparatus? Entropy-Based Topic Modeling on Multiple Domain-Specific Text Collections},
    year = {2018}
    }
Please also note our earlier short paper [**What Should I Cite? Cross-Collection Reference Recommendation of Patents and Papers**](https://hpi.de/fileadmin/user_upload/fachgebiete/naumann/people/risch/risch2017what.pdf) on a related topic published at the International Conference on Theory and Practice of Digital Libraries (TDPL).

## Implementation
* `TopicModelCcLDA.java` implements entity-based cross-collection latent Dirichlet allocation
* `RunTopicModel.java` starts the training of the topic model and the following evalution
* `CorpusBlogPostsFromFile.java` loads a corpus from a file.
* `CorpusToy.java` loads a small example corpus of a few documents defined in the code.
