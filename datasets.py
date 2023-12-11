import numpy as np
import os
import pickle
import torch.utils.data


class PlainDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_name, data_path, mode='train'):
        super(PlainDataset, self).__init__()
        print('\n==> Loading dataset...')
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        if dataset_name in ['20ng', 'dbpedia', 'wos', 'yahoo']:
            train_corpus = data['base_bows'].toarray().T
            test_corpus = data['novel_bows'].toarray().T
            vocabulary = data['vocab']
        else:
            raise NotImplementedError(f'unknown dataset: {dataset_name}')

        if mode == 'train':
            shuffle_ids = np.random.permutation(train_corpus.shape[0])
            self.data = train_corpus[shuffle_ids]
        else:
            self.data = test_corpus

        self.vocab = vocabulary
        print('Done, %s set information:' % mode)
        print('Num of documents: %d' % self.data.shape[0])
        print('Num of vocabulary terms: %d' % len(vocabulary))

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index]).float()

    def __len__(self):
        return self.data.shape[0]


def sup_qry_split(bows, query_ratio):
    support_bows = np.zeros_like(bows)
    query_bows = np.zeros_like(bows)
    for doc_id in range(bows.shape[0]):
        doc = bows[doc_id]
        nonzero_ids = np.nonzero(doc)[0]
        for word_id in nonzero_ids:
            count = doc[word_id]
            for _ in range(int(count)):
                R = np.random.binomial(n=1, p=query_ratio, size=1)[0]
                if R == 1:
                    query_bows[doc_id, word_id] += 1.
                else:
                    support_bows[doc_id, word_id] += 1.
    return support_bows, query_bows


class MetaDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_name, data_path, mode, task_size, query_ratio=0.2):
        super(MetaDataset, self).__init__()
        print('\n==> Loading dataset...')
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        if dataset_name in ['20ng', 'dbpedia', 'wos', 'yahoo']:
            vocabulary = data['vocab']
            train_class_id = data['base_class_id']
            test_class_id = data['novel_class_id']
        else:
            raise NotImplementedError(f'unknown dataset: {dataset_name}')

        self.vocab = vocabulary
        self.class_id = train_class_id if mode == 'train' else test_class_id
        print('Done, %s set information:' % mode)
        print('Number of corpora with different topics: %d' % len(self.class_id))
        print('Number of vocabulary terms: %d' % len(self.vocab))

        self.task_paths = []
        self.task_size = task_size
        self.query_ratio = query_ratio

        task_dir = './data/{}/meta_{}/task_size_{}'.format(dataset_name, mode, task_size)
        for _, _, files in os.walk(task_dir):
            for name in files:
                self.task_paths.append(os.path.join(task_dir, name))

    def __getitem__(self, index):
        with open(self.task_paths[index], 'rb') as f:
            task_data = pickle.load(f)
        all_bow = task_data['bow'].toarray().T
        sup_bow, qry_bow = sup_qry_split(all_bow, self.query_ratio)
        adj_mat = task_data['dep_graph'].toarray()
        ind_mat = task_data['word_indicator'].toarray()
        # return torch.from_numpy(sup_bow).float(), torch.from_numpy(qry_bow).float()
        # ToDo: only our Meta-CETM needs to return extra "adj_mat", "ind_mat", and "topic"
        #  For other methods, please use the above commented line
        return torch.from_numpy(sup_bow).float(), torch.from_numpy(qry_bow).float(), \
               torch.from_numpy(adj_mat).float(), torch.from_numpy(ind_mat).float(), task_data['topic']

    def __len__(self):
        return len(self.task_paths)


class PlainClsDataset(torch.utils.data.Dataset):
    """This is prepared for few-shot text classification.
    """

    def __init__(self, dataset_name, data_path, mode='train'):
        super(PlainClsDataset, self).__init__()
        print('\n==> Loading dataset...')
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        if dataset_name in ['20ng', 'dbpedia', 'wos', 'yahoo']:
            train_corpus = data['base_bows'].toarray().T
            train_labels = data['base_labels']
            test_corpus = data['novel_bows'].toarray().T
            test_labels = data['novel_labels']

            vocabulary = data['vocab']
            train_class_id = data['base_class_id']
            test_class_id = data['novel_class_id']
        else:
            raise NotImplementedError(f'unknown dataset: {dataset_name}')

        if mode == 'train':
            shuffle_ids = np.random.permutation(train_corpus.shape[0])
            self.data = train_corpus[shuffle_ids]
            self.target = train_labels[shuffle_ids]
        else:
            self.data = test_corpus
            self.target = test_labels

        self.vocab = vocabulary
        self.class_id = train_class_id if mode == 'train' else test_class_id
        self.cls2lab = {self.class_id[l]: l for l in range(len(self.class_id))}
        print('Done, %s set information:' % mode)
        print('Num of documents: %d' % self.data.shape[0])
        print('Num of vocabulary terms: %d' % len(vocabulary))

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index]).float(), self.cls2lab[self.target[index]]

    def __len__(self):
        return self.data.shape[0]


class MetaClsDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_name, data_path, mode, num_ways, num_shots):
        super(MetaClsDataset, self).__init__()
        print('\n==> Loading dataset...')
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        if dataset_name in ['20ng', 'dbpedia', 'yahoo', 'wos']:
            vocabulary = data['vocab']
            train_class_id = data['base_class_id']
            test_class_id = data['novel_class_id']
        else:
            raise NotImplementedError(f'unknown dataset: {dataset_name}')

        self.vocab = vocabulary
        self.class_id = train_class_id if mode == 'train' else test_class_id
        print('Done, %s set information:' % mode)
        print('Number of corpora with different topics: %d' % len(self.class_id))
        print('Number of vocabulary terms: %d' % len(self.vocab))

        self.task_paths = []
        self.num_ways = num_ways
        self.num_shots = num_shots
        task_dir = './data/{}/meta_{}/{}_way_{}_shot'.format(dataset_name, mode, num_ways, num_shots)
        for _, _, files in os.walk(task_dir):
            for name in files:
                self.task_paths.append(os.path.join(task_dir, name))

    def __getitem__(self, index):
        with open(self.task_paths[index], 'rb') as f:
            task_data = pickle.load(f)
        sup_sets, qry_sets = task_data['sup_batch'], task_data['qry_batch']
        adj_list, ind_list = task_data['adj_batch'], task_data['ind_batch']
        return torch.from_numpy(sup_sets).float(), torch.from_numpy(qry_sets).float()
        # ToDo: only our Meta-CETM needs to return extra "adj_mat", "ind_mat", and "topic"
        #  For other methods, please use the above commented line
        # return torch.from_numpy(sup_sets).float(), torch.from_numpy(qry_sets).float(), adj_list, ind_list

    def __len__(self):
        return len(self.task_paths)


if __name__ == '__main__':
    # train_set = MetaDataset('20ng', './data/20ng/20ng_8novel.pkl', 'train', 5, 0.2)
    # sup, qry, adj, ind, name = train_set.__getitem__(7)
    # print('Number of total tasks:', len(train_set))
    # print('\n==> The 7-th task info:')
    # print('Shape of support set:', sup.shape)
    # print('Shape of query set:', qry.shape)
    # print('Shape of adjacency matrix:', adj.shape)
    # print('Shape of indicator matrix:', ind.shape)

    train_set = MetaClsDataset('20ng', './data/20ng/20ng_8novel.pkl', 'train', 5, 5)
    sup, qry, adj, ind = train_set.__getitem__(7)
    print('Number of total tasks:', len(train_set))
    print('\n==> The 7-th task info:')
    print('Shape of support set:', sup.shape)
    print('Shape of query set:', qry.shape)
    for j, A in enumerate(adj):
        print(f'Shape of {j}-th adjacency matrix:', A.shape)
    for j, I in enumerate(ind):
        print(f'Shape of {j}-th indicator matrix:', I.shape)
