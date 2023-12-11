import numpy as np
import os
import pickle
import spacy
import time
from tqdm import tqdm
from scipy import sparse

nlp = spacy.load("en_core_web_sm")


def build_dep_graph(bows, docs, vocab):
    word_ids = np.nonzero(bows.toarray().sum(1))[0]
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
num_train_tasks = 12000
num_test_tasks = 1200
num_val_tasks = 1200
task_size = 10

print('==> Loading dataset...')
with open(data_path, 'rb') as f:
    data = pickle.load(f)

if dataset_name in ['20ng', 'dbpedia', 'wos', 'yahoo']:
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

# Construct training tasks
print('\n==> Sampling a batch of training tasks...')
t0 = time.time()
np.random.seed(2023)
for i in tqdm(range(num_train_tasks), desc='Processing'):
    topic_id = np.random.choice(train_class_id)
    sample_ids = np.random.choice(
        len(cls2data[topic_id]['docs']), task_size, replace=False
    )
    sampled_bows = cls2data[topic_id]['bows'][:, sample_ids]
    sampled_docs = np.array(
        cls2data[topic_id]['docs'], dtype=object)[sample_ids].tolist()
    adj_mat, ind_mat = build_dep_graph(sampled_bows, sampled_docs, vocabulary)

    save_dir = f'../data/{dataset_name}/meta_train/task_size_{task_size}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, f'task{i}.pkl'), 'wb') as f:
        pickle.dump({'topic': topic_names[topic_id],
                     'bow': sampled_bows,
                     'dep_graph': adj_mat,
                     'word_indicator': ind_mat}, f)
print("Meta-train set processed, done in %0.3fs." % (time.time() - t0))

# Construct validation tasks
print('\n==> Sampling a batch of validation tasks...')
t0 = time.time()
np.random.seed(2023)
for j in tqdm(range(num_val_tasks), desc='Processing'):
    topic_id = np.random.choice([7, 9, 11, 13])  # 20ng
    # topic_id = np.random.choice([10, 11])        # dbpedia
    # topic_id = 6                                 # wos
    # topic_id = np.random.choice([2, 9])          # yahoo
    sample_ids = np.random.choice(
        len(cls2data[topic_id]['docs']), task_size, replace=False
    )
    sampled_bows = cls2data[topic_id]['bows'][:, sample_ids]
    sampled_docs = np.array(
        cls2data[topic_id]['docs'], dtype=object)[sample_ids].tolist()
    adj_mat, ind_mat = build_dep_graph(sampled_bows, sampled_docs, vocabulary)

    save_dir = f'../data/{dataset_name}/meta_val/task_size_{task_size}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, f'task{j}.pkl'), 'wb') as f:
        pickle.dump({'topic': topic_names[topic_id],
                     'bow': sampled_bows,
                     'dep_graph': adj_mat,
                     'word_indicator': ind_mat}, f)
print("Meta-val set processed, done in %0.3fs." % (time.time() - t0))

# Construct test tasks
print('\n==> Sampling a batch of test tasks...')
t0 = time.time()
np.random.seed(2023)
for k in tqdm(range(num_test_tasks), desc='Processing'):
    topic_id = np.random.choice([8, 10, 12, 14])  # 20ng
    # topic_id = np.random.choice([8, 9, 12, 13])    # dbpedia
    # topic_id = np.random.choice([2, 5])            # wos
    # topic_id = np.random.choice([4, 5])            # yahoo
    sample_ids = np.random.choice(
        len(cls2data[topic_id]['docs']), task_size, replace=False
    )
    sampled_bows = cls2data[topic_id]['bows'][:, sample_ids]
    sampled_docs = np.array(
        cls2data[topic_id]['docs'], dtype=object)[sample_ids].tolist()
    adj_mat, ind_mat = build_dep_graph(sampled_bows, sampled_docs, vocabulary)

    save_dir = f'../data/{dataset_name}/meta_test/task_size_{task_size}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, f'task{k}.pkl'), 'wb') as f:
        pickle.dump({'topic': topic_names[topic_id],
                     'bow': sampled_bows,
                     'dep_graph': adj_mat,
                     'word_indicator': ind_mat}, f)
print("Meta-test set processed, done in %0.3fs." % (time.time() - t0))
