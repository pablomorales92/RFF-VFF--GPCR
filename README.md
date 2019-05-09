Code for the methods RFFGPCR and VFFGPCR, which leverage Fourier features approximations to scale up GP-based crowdsourcing models. Full reference:

Morales-Álvarez P., Ruiz P., Santos-Rodríguez R., Molina R., Katsaggelos A.K.\
Scalable and efficient learning from crowds with Gaussian processes\
Information Fusion, 2019\
DOI: https://doi.org/10.1016/j.inffus.2018.12.008

## Abstract
Over the last few years, multiply-annotated data has become a very popular source of information. Online platforms such as Amazon Mechanical Turk have revolutionized the labelling process needed for any classification task, sharing the effort between a number of annotators (instead of the classical single expert). This crowdsourcing approach has introduced new challenging problems, such as handling disagreements on the annotated samples or combining the unknown expertise of the annotators. Probabilistic methods, such as Gaussian Processes (GP), have proven successful to model this new crowdsourcing scenario. However, GPs do not scale up well with the training set size, which makes them prohibitive for medium-to-large datasets (beyond 10K training instances). This constitutes a serious limitation for current real-world applications. In this work, we introduce two scalable and efficient GP-based crowdsourcing methods that allow for processing previously-prohibitive datasets. The first one is an efficient and fast approximation to GP with squared exponential (SE) kernel. The second allows for learning a more flexible kernel at the expense of a heavier training (but still scalable to large datasets). Since the latter is not a GP-SE approximation, it can be also considered as a whole new scalable and efficient crowdsourcing method, useful for any dataset size. Both methods use Fourier features and variational inference, can predict the class of new samples, and estimate the expertise of the involved annotators. A complete experimentation compares them with state-of-the-art probabilistic approaches in synthetic and real crowdsourcing datasets of different sizes. They stand out as the best performing approach for large scale problems. Moreover, the second method is competitive with the current state-of-the-art for small datasets.

## Citation
@article{morales2019scalable,\
  title={Scalable and efficient learning from crowds with Gaussian processes},\
  author={Morales-{\\'A}lvarez, Pablo and Ruiz, Pablo and Santos-Rodr{\\'i}guez, Ra{\\'u}l and Molina, Rafael and Katsaggelos, Aggelos K},\
  journal={Information Fusion},\
  volume={52},\
  pages={110--127},\
  year={2019},\
  publisher={Elsevier}\
}
