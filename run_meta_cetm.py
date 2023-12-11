import argparse
import numpy as np
import os
from time import time

import torch
import torch.nn as nn
import torch.utils.data

from datasets import MetaDataset
from models import MetaContextETM
from utils.train_util import load_glove_embeddings, visualize_topics


parser = argparse.ArgumentParser(description='Combining graph contextual information with ETM for few-shot learning')

# data-related arguments
parser.add_argument('--seed', type=int, default=2023, help='random seed.')
parser.add_argument('--dataset', type=str, default='20ng', help='name of used dataset.')
parser.add_argument('--data_path', type=str, default='./data/20ng/20ng_8novel.pkl', help='path to load raw data file.')
parser.add_argument('--embed_path', type=str, default='./data/glove.6B/glove.6B.100d.txt', help='path of pretrained glove embeddings.')
parser.add_argument('--save_dir', type=str, default='./results', help='directory to save visualization results.')

# model-related arguments
parser.add_argument('--vocab_size', type=int, default=5968,
                    help='number of unique terms in the vocabulary.')
parser.add_argument('--embed_size', type=int, default=100,
                    help='dimensionality of the word embedding space.')
parser.add_argument('--num_topics', type=int, default=20,
                    help='number of topics to be adapted from a given task.')
parser.add_argument('--num_hiddens', type=int, default=300,
                    help='number of hidden units for the BoW encoder q(theta).')
parser.add_argument('--act', type=str, default='relu',
                    help='activation function, could be: tanh, softplus, relu, elu.')
parser.add_argument('--dropout_rate', type=float, default=0.0,
                    help='dropout rate of encoder for q(theta).')
parser.add_argument('--graph_h_dim', type=int, default=100,
                    help='number of hidden units for the graph encoder.')
parser.add_argument('--z_dim', type=int, default=32,
                    help='dimension of adapte word embeddings by GCN.')
parser.add_argument('--fix_pi', type=bool, default=False,
                    help='use a fixed coefficient in the Gaussian mixture prior.')
parser.add_argument('--em_iterations', type=int, default=100,
                    help='number of EM steps to estimate Gaussian mixture parameters.')
parser.add_argument('--train_mc_sample_size', type=int, default=5,
                    help='sampling size for Monte Carlo approximation.')

# optimization-related arguments
parser.add_argument('--mode', type=str, default='train',
                    help='train or eval model.')
parser.add_argument('--load_from', type=str, default='',
                    help='path to the checkpoint for eval purpose.')
parser.add_argument('--optimizer', type=str, default='adam',
                    help='choice of optimizer.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate.')
parser.add_argument('--grad_clip', type=float, default=20.0,
                    help='gradient clipping for stable training.')
parser.add_argument('--weight_decay', type=float, default=1.2e-6, help='weight decay for l2 regularization.')
parser.add_argument('--lr_factor', type=float, default=4.0, help='divide learning rate by this...')
parser.add_argument('--anneal_lr', type=int, default=0, help='whether to anneal the learning rate or not.')

# few-shot setting arguments
parser.add_argument('--docs_per_task', type=int, default=10,
                    help='number of documents for each individual task.')
parser.add_argument('--heldout-rate', type=float, default=0.2,
                    help='proportion of remaining word tokens used to calculate perplexity.')
args = parser.parse_args()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # data loading pipeline
    train_set = MetaDataset(
        dataset_name=args.dataset,
        data_path=args.data_path,
        mode='train',
        task_size=args.docs_per_task,
        query_ratio=args.heldout_rate
    )
    val_set = MetaDataset(
        dataset_name=args.dataset,
        data_path=args.data_path,
        mode='val',
        task_size=args.docs_per_task,
        query_ratio=args.heldout_rate
    )
    test_set = MetaDataset(
        dataset_name=args.dataset,
        data_path=args.data_path,
        mode='test',
        task_size=args.docs_per_task,
        query_ratio=args.heldout_rate
    )
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)

    # adjust the input size
    vocabulary = train_set.vocab
    args.vocab_size = len(vocabulary)

    # initialize word embeddings with pretrained glove embeddings
    pretrained_embeddings = None
    if args.embed_path:
        print('\n==> Loading pretrained word embeddings...')
        pretrained_embeddings = load_glove_embeddings(args.embed_path, vocabulary)
        args.embed_size = pretrained_embeddings.shape[1]

    # define model and meta learner
    model = MetaContextETM(args, device, pretrained_embeddings)
    model.to(device)
    print('\nModel : {}'.format(model))
    tmp = filter(lambda x: x.requires_grad, model.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print('Total trainable parameters:', num)

    optimizer = torch.optim.Adam(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    if args.mode == 'train':
        best_val_ppl = 1e9

        print('\n===> Meta-training stage ===<')
        for epoch in range(6):
            model.train()
            t_start = time()
            loss_list, bll_list, gll_list, klc_list, kl_theta_list, klz_list = [], [], [], [], [], []
            for i, (support_data, query_data, adj, ind, _) in enumerate(train_loader):
                # support_data, query_data = support_data.to(device), query_data.to(device)
                task_data = (support_data + query_data).squeeze(0).to(device)
                adj = adj.squeeze(0).to(device)
                ind = ind.squeeze(0).to(device)

                neg_elbo, neg_bow_ll, neg_graph_ll, c_kl, theta_kl, z_kl, _ = model(task_data, adj, ind)
                loss_list.append(neg_elbo.item())
                bll_list.append(neg_bow_ll.item())
                gll_list.append(neg_graph_ll.item())
                klc_list.append(c_kl.item())
                kl_theta_list.append(theta_kl.item())
                klz_list.append(z_kl.item())

                # optimize the model parameters
                flag = 0
                for para in model.parameters():
                    flag += torch.sum(torch.isnan(para))
                if flag == 0:
                    optimizer.zero_grad()
                    neg_elbo.backward()
                    if args.grad_clip > 0:
                        for param in list(model.parameters()):
                            nn.utils.clip_grad_norm_(param, args.grad_clip)
                    optimizer.step()

                # print the loss information
                if (i + 1) % 100 == 0:
                    print('Epoch: [{}/{}] \tLoss: {:.8f} \tNeg_bow_ll: {:.8f} \tNeg_graph_ll: {:.8f} '
                          '\tKL_c: {:.8f} \tKL_theta: {:.8f}\t KL_z: {:.8f}'.format(
                        i + 1, epoch + 1, np.mean(loss_list), np.mean(bll_list), np.mean(gll_list),
                        np.mean(klc_list), np.mean(kl_theta_list), np.mean(klz_list))
                    )

                # evaluation on the validation set
                if (i + 1) % 2000 == 0:
                    model.eval()
                    print('\nEvaluate model on validation set...')
                    task_id = 0
                    all_val_ppls = []
                    for (support_set, query_set, adj, ind, gt_topic) in val_loader:
                        support_set = support_set.squeeze(0).to(device)
                        query_set = query_set.squeeze(0).to(device)
                        adj = adj.squeeze(0).to(device)
                        ind = ind.squeeze(0).to(device)
                        with torch.no_grad():
                            pred, topic_matrix = model.predict(support_set, adj, ind)
                            # calculate per-holdout-word perplexity
                            val_ppl = model.get_ppl(query_set.t(), pred.t())
                            all_val_ppls.append(val_ppl.item())

                        # visualize the content of mined topics by reading the tea leaves
                        task_id = task_id + 1
                        val_count = (i + 1) // 2000
                        save_dir = os.path.join(args.save_dir, f'{args.dataset}/Meta_CETM/{args.docs_per_task}shot_epoch{epoch+1}_val{val_count}')
                        visualize_topics(topic_matrix, save_dir, vocabulary, top_n=20, mark=str(task_id) + '_' + gt_topic[0])

                    # save the best perplexity result
                    ppl_mean, ppl_std = np.array(all_val_ppls).mean(axis=0), np.array(all_val_ppls).std(axis=0)
                    print("The average perplexity on {} val tasks: {:.6f} \u00B1 {:.6f}".format(
                        len(val_loader), ppl_mean, ppl_std)
                    )
                    if ppl_mean < best_val_ppl:
                        best_val_ppl = ppl_mean
                        os.makedirs('./checkpoints/Meta_CETM', exist_ok=True)
                        torch.save(
                            model.state_dict(),
                            f'./checkpoints/Meta_CETM/{args.dataset}_{args.docs_per_task}shot_best_val_ppl.pth'
                        )

                    # save the trained model weights
                    # os.makedirs('./checkpoints/Meta_CETM', exist_ok=True)
                    # torch.save(
                    #     model.state_dict(),
                    #     f'./checkpoints/Meta_CETM/{args.dataset}_{args.docs_per_task}shot_epoch{epoch + 1}_step{i + 1}.pth'
                    # )
                    model.train()

            print("Epoch {} finished in {:.4f}s".format(epoch + 1, (time() - t_start)))
        print("Training finished, the best validation ppl is: {}".format(best_val_ppl))

    else:
        print('\n==> Meta-test stage ===<')
        ckpt_path = f'./checkpoints/Meta_CETM/{args.dataset}_{args.docs_per_task}shot_best_val_ppl.pth',
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        task_id = 0
        all_test_ppls = []
        for (support_set, query_set, adj, ind, gt_topic) in test_loader:
            support_set = support_set.squeeze(0).to(device)
            query_set = query_set.squeeze(0).to(device)
            adj = adj.squeeze(0).to(device)
            ind = ind.squeeze(0).to(device)

            with torch.no_grad():
                pred, topic_matrix = model.predict(support_set, adj, ind)
                test_ppl = model.get_ppl(query_set.t(), pred.t())
                all_test_ppls.append(test_ppl.item())

            task_id = task_id + 1
            visualize_topics(
                topic_matrix,
                './results/{}/Meta_CETM/test_{}shot'.format(args.dataset, args.docs_per_task),
                vocabulary,
                top_n=adj.size(0) // args.num_topics,
                mark=str(task_id) + '_' + gt_topic[0]
            )

        ppl_mean, ppl_std = np.array(all_test_ppls).mean(axis=0), np.array(all_test_ppls).std(axis=0)
        print("The average perplexity on {} test tasks: {:.6f} \u00B1 {:.6f}".format(
            len(test_loader), ppl_mean, ppl_std))


if __name__ == '__main__':
    main()
