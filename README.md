# Meta-CETM

This is the official implementation for the NeurIPS 2023 paper 'Context-guided Embedding Adaptation for Effective Topic Modeling in Low-Resource Regimes'

## Get started
The following lists the statistics of the datasets we used.
| Dataset | Source link | *N* (#docs) | *V* (#words) | *L* (#labels)
| :----- | :-----: | :-----: | :-----: | :-----: |
|*20Newsgroups* | [20NG](http://qwone.com/~jason/20Newsgroups/) | 11288 | 5968 | 20 |
|*Yahoo! Answers* | [Yahoo](https://github.com/LC-John/Yahoo-Answers-Topic-Classification-Dataset) | 27069 | 7507 | 10 |
|*DBpedia* | [DB14](https://www.dbpedia.org/) | 30183 | 6274 | 14 |
|*Web of Science* | [WOS](https://data.mendeley.com/datasets/9rw3vkcfy4/6) | 11921 | 4923 | 7 |

We curated the vocabulary for each dataset by removing those words with very low and very high frequencies, as well as a list of commonly used stop words. After that, we filtered out documents that contained less than 50 vocabulary terms to yield the final available part of each original dataset. The pre-processed version of all four datasets can be downloaded from
- https://drive.google.com/file/d/1byla0PKb27HXadonut_qf7OVAYEwauhX/view?usp=drive_link


## Episodic task construction
