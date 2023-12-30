# Meta-CETM

This is the official implementation for the NeurIPS 2023 paper ***Context-guided Embedding Adaptation for Effective Topic Modeling in Low-Resource Regimes***. We have developed an approach that is proficient in discovering meaningful topics from only a few documents, and the core idea is to adaptively generate word embeddings semantically tailored to the given task by fully exploiting the contextual syntactic information. <!--We hope this will offer an alternative for the text analysis under low-resource scenarios.-->

<img src="/display/overview.png" width="672" height="320">

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
Since we adopted an episodic training strategy to learn our model, we need to sample a batch of tasks from the original corpus to construct the training, validation, and test sets separately. To do this, `unzip` the downloaded pre-processed datasets, put the `data` folder under the root directory, and then execute the following command.
```bash
cd utils
python process_to_task.py
```
Note that for different datasets, please modify the arguments **dataset_name** and **data_path** accordingly.

## Experiment: per-holdout-word perplexity (PPL)
To train a **Meta-CETM** with the best predictive performance from scratch, run the following command
```bash
python run_meta_cetm.py --dataset 20ng --data_path ./data/20ng/20ng_8novel.pkl --embed_path ./data/glove.6B/glove.6B.100d.txt --docs_per_task 10 --num_topics 20 --mode train
```
To train a **ETM** using the model-agnostic meta-learning ([MAML](https://arxiv.org/abs/1703.03400)) strategy, run the following command
```bash
python run_etm.py --dataset 20ng --data_path ./data/20ng/20ng_8novel.pkl --embed_path ./data/glove.6B/glove.6B.100d.txt --docs_per_task 10 --num_topics 20 --mode train --maml_train True
```
In the same vein, to train a **ProdLDA** from scratch using *MAML*, you can run the command
```bash
python run_avitm.py --dataset 20ng --data_path ./data/20ng/20ng_8novel.pkl --docs_per_task 10 --num_topics 20 --mode train --maml_train True
```
