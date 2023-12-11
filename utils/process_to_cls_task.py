import numpy as np
import os
import pickle
import spacy
import time
from tqdm import tqdm
from scipy import sparse

nlp = spacy.load("en_core_web_sm")


def build_dep_graph(bows, docs, vocab):
    word_ids = np.nonzero(bows.sum(1))[0]
    V = len(word_ids)
    dependency_graph = np.eye(V)
    for doc in nlp.pipe(docs, n_process=4):
        for token in doc:
            if token.dep_ in ["compound", "amod", "conj", "dobj", "relcl", "nummod"]:
                try:
                    m = vocab.index(token.text)
                    n = vocab.index(token.head.text)
                    dependency_graph[np.where(word_ids == m)[0], np.where(word_ids == n)[0]] = 1
                    dependency_graph[np.where(word_ids == n)[0], np.where(word_ids == m)[0]] = 1
                    # print('{} <--> {}, {}'.format(token.text, token.head.text, token.dep_))
                except:
                    continue
    return sparse.csc_matrix(dependency_graph), sparse.csc_matrix(np.eye(len(vocab))[word_ids])


dataset_name = '20ng'
data_path = '../data/20ng/20ng_8novel.pkl'
num_train_tasks = 60
num_test_tasks = 10
n_ways = 5
n_shots = 5
n_queries = 15

print('==> Loading dataset...')
with open(data_path, 'rb') as f:
    data = pickle.load(f)

if dataset_name in ['20ng', 'dbpedia', 'yahoo', 'wos']:
    vocabulary = data['vocab']
    cls2data = data['label2data']
    topic_names = data['topic_names']
    train_class_id = data['base_class_id']
    test_class_id = data['novel_class_id']

    print('Done, dataset information:')
    print('Number of training corpora with different topics: %d' % len(train_class_id))
    print('Number of test corpora with different topics: %d' % len(test_class_id))
    print('Number of vocabulary terms: %d' % len(vocabulary))
else:
    raise NotImplementedError(f'unknown dataset: {dataset_name}')

# Construct training tasks for few-shot text classification experiment
print('\n==> Sampling a batch of training tasks...')
t0 = time.time()
np.random.seed(2023)
for i in tqdm(range(num_train_tasks), desc='Processing'):
    selected_cls = np.random.choice(train_class_id, n_ways, replace=False)
    sup_batch, qry_batch = [], []
    adj_batch, ind_batch = [], []
    for c in selected_cls:
        sample_ids = np.random.choice(
            len(cls2data[c]['docs']), n_shots + n_queries, replace=False
        )
        support_ids, query_ids = sample_ids[: n_shots], sample_ids[n_shots:]
        support_bows = cls2data[c]['bows'][:, support_ids].toarray()
        query_bows = cls2data[c]['bows'][:, query_ids].toarray()
        support_docs = np.array(
            cls2data[c]['docs'], dtype=object)[support_ids].tolist()
        adj_mat, ind_mat = build_dep_graph(support_bows, support_docs, vocabulary)
        sup_batch.append(support_bows.T)
        qry_batch.append(query_bows.T)
        adj_batch.append(adj_mat)
        ind_batch.append(ind_mat)
    sup_batch, qry_batch = np.stack(sup_batch), np.stack(qry_batch)

    save_dir = f'../data/{dataset_name}/meta_train/{n_ways}_way_{n_shots}_shot'
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f'task{i}.pkl'), 'wb') as f:
        pickle.dump({'sup_batch': sup_batch,
                     'qry_batch': qry_batch,
                     'adj_batch': adj_batch,
                     'ind_batch': ind_batch}, f)
print("Meta-train set processed, done in %0.3fs." % (time.time() - t0))

# Construct test tasks for few-shot text classification experiment
print('\n==> Sampling a batch of test tasks...')
t0 = time.time()
np.random.seed(2023)
for j in tqdm(range(num_test_tasks), desc='Processing'):
    selected_cls = np.random.choice(test_class_id, n_ways, replace=False)
    sup_batch, qry_batch = [], []
    adj_batch, ind_batch = [], []
    for c in selected_cls:
        sample_ids = np.random.choice(
            len(cls2data[c]['docs']), n_shots + n_queries, replace=False
        )
        support_ids, query_ids = sample_ids[: n_shots], sample_ids[n_shots:]
        support_bows = cls2data[c]['bows'][:, support_ids].toarray()
        query_bows = cls2data[c]['bows'][:, query_ids].toarray()
        support_docs = np.array(
            cls2data[c]['docs'], dtype=object)[support_ids].tolist()
        adj_mat, ind_mat = build_dep_graph(support_bows, support_docs, vocabulary)
        sup_batch.append(support_bows.T)
        qry_batch.append(query_bows.T)
        adj_batch.append(adj_mat)
        ind_batch.append(ind_mat)
    sup_batch, qry_batch = np.stack(sup_batch), np.stack(qry_batch)

    save_dir = f'../data/{dataset_name}/meta_test/{n_ways}_way_{n_shots}_shot'
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f'task{j}.pkl'), 'wb') as f:
        pickle.dump({'sup_batch': sup_batch,
                     'qry_batch': qry_batch,
                     'adj_batch': adj_batch,
                     'ind_batch': ind_batch}, f)
print("Meta-test set processed, done in %0.3fs." % (time.time() - t0))
