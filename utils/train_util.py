import numpy as np
import os
from time import time


def load_glove_embeddings(embed_path, vocab):
    """Initial word embeddings with pretrained glove embeddings if necessary.
    """
    embed_size = int(embed_path.split('.')[-2][:-1])
    print('Loading pretrained glove embeddings...')
    t0 = time()
    embeddings_dict = dict()
    with open(embed_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            values = line.split()
            word = values[0]
            embedding = np.asarray(values[1:], dtype=np.float32)
            embeddings_dict[word] = embedding
    print("Done in %0.3fs." % (time() - t0))

    print('Initialize word embeddings with glove embeddings...')
    t0 = time()
    vocab_embeddings = list()
    for word in vocab:
        try:
            vocab_embeddings.append(embeddings_dict[word])
        except:
            vocab_embeddings.append(np.random.randn(embed_size))
    print("Done in %0.3fs." % (time() - t0))

    return np.array(vocab_embeddings, dtype=np.float32)


def topic_diversity(topic_matrix, top_k=25):
    """ Topic Diversity (TD) measures how diverse the discovered topics are.

    We define topic diversity to be the percentage of unique words in the top 25 words (Dieng et al., 2020)
    of the selected topics. TD close to 0 indicates redundant topics, TD close to 1 indicates more varied topics.

    Args:
        topic_matrix: shape [K, V]
        top_k:
    """
    num_topics = topic_matrix.shape[0]
    top_words_idx = np.zeros((num_topics, top_k))
    for k in range(num_topics):
        idx = np.argsort(topic_matrix[k, :])[::-1][:top_k]
        top_words_idx[k, :] = idx
    num_unique = len(np.unique(top_words_idx))
    num_total = num_topics * top_k
    td = num_unique / num_total
    # print('Topic diversity is: {}'.format(td))
    return td


def get_top_n(phi_column, vocab, top_n=25):
    top_n_words = ''
    indices = np.argsort(-phi_column)
    for n in range(top_n):
        top_n_words += vocab[indices[n]]
        top_n_words += ' '
    return top_n_words


def visualize_topics(phis, save_dir, vocab, top_n=25, mark=None):
    if isinstance(phis, list):
        phis = [phi.cpu().numpy() for phi in phis]
    else:
        phis = [phis.cpu().numpy()]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    factorial_phi = 1
    for layer_id, phi in enumerate(phis):
        factorial_phi = np.dot(factorial_phi, phi)
        cur_td = topic_diversity(factorial_phi.T, top_n)

        num_topics = factorial_phi.shape[1]
        if mark is not None:
            path = os.path.join(save_dir, 'task' + str(mark) + '_phi' + str(layer_id) + '.txt')
        else:
            path = os.path.join(save_dir, 'phi' + str(layer_id) + '.txt')
        f = open(path, 'w')
        for k in range(num_topics):
            top_n_words = get_top_n(
                factorial_phi[:, k], vocab, top_n)
            f.write(top_n_words)
            f.write('\n')
        f.write('Topic diversity:{}'.format(cur_td))
        f.write('\n')
        f.close()
