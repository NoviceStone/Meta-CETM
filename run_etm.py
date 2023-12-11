import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.utils.data

from copy import deepcopy
from time import time

from datasets import PlainDataset, MetaDataset
from models import ETM, MetaLearner
from utils.train_util import load_glove_embeddings


parser = argparse.ArgumentParser(description='run ETM for few-shot learning')

# data-related arguments
parser.add_argument('--seed', type=int, default=2023, help='random seed')
parser.add_argument('--dataset', type=str, default='', help='name of corpus')
parser.add_argument('--data_path', type=str, default='', help='path to load dataset')
parser.add_argument('--embed_path', type=str, default='', help='path to pretrained word embeddings')
parser.add_argument('--save_dir', type=str, default='./results', help='directory to save results')
parser.add_argument('--batch_size', type=int, default=200, help='input batch size for training')

# model-related arguments
parser.add_argument('--vocab_size', type=int, default=5968, help='number of words in the vocabulary')
parser.add_argument('--num_topics', type=int, default=20, help='number of topics to be mined')
parser.add_argument('--embed_size', type=int, default=100, help='dimensionality of word embedding space')
parser.add_argument('--num_hiddens', type=int, default=300, help='number of hidden units of encoder for q(theta)')
parser.add_argument('--act', type=str, default='relu', help='(tanh, softplus, relu, leakyrelu, elu, selu)')
parser.add_argument('--dropout_rate', type=float, default=0.0, help='dropout rate of encoder for q(theta)')

# optimization-related arguments
parser.add_argument('--mode', type=str, default='train', help='training phase or testing phase')
parser.add_argument('--maml_train', type=bool, default=False, help='use the meta-training strategy')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs for pretraining strategy')
parser.add_argument('--optimizer', type=str, default='adam', help='choice of optimizer')
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--grad_clip', type=float, default=20.0, help='gradient clipping')
parser.add_argument('--weight_decay', type=float, default=1.2e-6, help='some l2 regularization')
parser.add_argument('--anneal_lr', type=int, default=1, help='whether to anneal the learning rate or not')

# few-shot setting arguments
parser.add_argument('--meta_batch_size', type=int, default=5, help='number of tasks processed at each update')
parser.add_argument('--meta_lr', type=float, default=5e-4, help='learning rate for meta-training outer loop')
parser.add_argument('--update_lr', type=float, default=5e-3, help='learning rate for meta-training inner loop')
parser.add_argument('--update_step', type=int, default=5, help='number of inner updated steps for meta-training')
parser.add_argument('--update_step_test', type=int, default=10, help='number of inner updated steps for meta-testing')
parser.add_argument('--docs_per_task', type=int, default=10, help='number of documents in each individual task')
parser.add_argument('--heldout-rate', type=float, default=0.2, help='proportion of remaining word tokens used to calculate perplexity')

args = parser.parse_args()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # data loading pipeline
    if args.maml_train:
        train_set = MetaDataset(
            dataset_name=args.dataset,
            data_path=args.data_path,
            mode='train',
            task_size=args.docs_per_task,
            query_ratio=args.heldout_rate
        )
        train_loader = None
    else:
        train_set = PlainDataset(args.dataset, args.data_path, 'train')
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, num_workers=4
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
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)

    # adjust the input size
    vocabulary = train_set.vocab
    args.vocab_size = len(vocabulary)

    # load pretrained word embeddings if necessary
    pretrained_embeddings = None
    if args.embed_path:
        print('\n==> Loading pretrained word embeddings...')
        pretrained_embeddings = load_glove_embeddings(args.embed_path, vocabulary)
        args.embed_size = pretrained_embeddings.shape[1]

    # define the base model
    model = ETM(args, device, pretrained_embeddings)
    model.to(device)
    print('\nModel : {}'.format(model))
    tmp = filter(lambda x: x.requires_grad, model.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print('Total trainable parameters:', num)

    if args.mode == 'train':
        best_val_ppl = 1e9

        # Meta-train and meta-test strategy
        if args.maml_train:
            learner = MetaLearner(args, model)
            print('\n===> Meta-training stage ===<')
            for epoch in range(6):
                train_loader = torch.utils.data.DataLoader(
                    train_set, batch_size=args.meta_batch_size, shuffle=True, num_workers=4)

                losses_on_qry = []
                for idx, (support_batch, query_batch) in enumerate(train_loader):
                    support_batch, query_batch = support_batch.to(device), query_batch.to(device)
                    loss_qry = learner(support_batch, query_batch)
                    losses_on_qry.append(loss_qry)
                    # losses_on_qry.append(0.)
                    if (idx + 1) % 20 == 0:
                        print('Epoch/Step: [{}/{}] \tAvgLoss on query set: {:.6f}'.format(
                            epoch + 1, (idx + 1) * args.meta_batch_size, np.mean(losses_on_qry)))

                    # eval on val set
                    if (idx + 1) % 400 == 0:
                        print("\n===> Meta-validation stage ===<")
                        all_val_ppls = []
                        for support_set, query_set in val_loader:
                            support_set, query_set = support_set.squeeze(0).to(device), query_set.squeeze(0).to(device)
                            val_ppl = learner.finetunning(support_set, query_set)
                            all_val_ppls.append(val_ppl.item())

                        ppl_mean, ppl_std = np.array(all_val_ppls).mean(axis=0), np.array(all_val_ppls).std(axis=0)
                        print("The average perplexity on {} val tasks: {:.6f} \u00B1 {:.6f}".format(
                            len(val_loader), ppl_mean, ppl_std)
                        )
                        if ppl_mean < best_val_ppl:
                            print("Achieving better perplexity: {:.8f}\n".format(ppl_mean))
                            best_val_ppl = ppl_mean
                            os.makedirs('./checkpoints/ETM_MAML', exist_ok=True)
                            torch.save(
                                learner.model.state_dict(),
                                os.path.join(
                                    './checkpoints/ETM_MAML',
                                    f'{args.dataset}_{args.docs_per_task}shot_best_val_ppl.pth'
                                )
                            )

        # Pretrain and Fine-tune strategy
        else:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay
            )
            if args.anneal_lr:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=int(0.75*args.epochs)
                )
            else:
                scheduler = None
            t_start = time()
            print('\n===> Pre-training stage ===<')
            for epoch in range(args.epochs):
                total_loss = []
                likelihood = []
                model.train()
                for idx, batch_data in enumerate(train_loader):
                    batch_data = batch_data.to(device)
                    nelbo, nll, kl, _ = model(batch_data)
                    total_loss.append(nelbo.item())
                    likelihood.append(nll.item())

                    flag = 0
                    for para in model.parameters():
                        flag += torch.sum(torch.isnan(para))
                    if flag == 0:
                        optimizer.zero_grad()
                        nelbo.backward()
                        if args.grad_clip > 0:
                            for param in list(model.parameters()):
                                nn.utils.clip_grad_norm_(param, args.grad_clip)
                        optimizer.step()

                    if (idx + 1) % 10 == 0:
                        print('Epoch: [{}/{}]\t Neg_ELBO: {}\t Neg_LL: {}'.format(
                            idx + 1, epoch + 1, np.mean(total_loss), np.mean(likelihood)))
                if scheduler is not None:
                    scheduler.step()

                # eval on val set
                if (epoch + 1) % 10 == 0:
                    print("\n===> Finetunning on validation set ===<")
                    all_val_ppls = []
                    for (support_set, query_set) in val_loader:
                        support_set = support_set.squeeze(0).to(device)
                        query_set = query_set.squeeze(0).to(device)

                        temp_model = deepcopy(model)
                        temp_optim = torch.optim.Adam(
                            temp_model.parameters(),
                            lr=args.update_lr
                        )
                        temp_model.train()
                        for _ in range(args.update_step_test):
                            loss, _, _, _ = temp_model(support_set)
                            temp_optim.zero_grad()
                            loss.backward()
                            temp_optim.step()
                        temp_model.eval()
                        with torch.no_grad():
                            _, _, _, pred = temp_model(support_set)
                            val_ppl = temp_model.get_ppl(query_set.t(), pred.t())
                            all_val_ppls.append(val_ppl.item())
                        del temp_model, temp_optim

                    ppl_mean, ppl_std = np.array(all_val_ppls).mean(axis=0), np.array(all_val_ppls).std(axis=0)
                    print("The average perplexity on {} val tasks: {:.6f} \u00B1 {:.6f}".format(
                        len(val_loader), ppl_mean, ppl_std)
                    )
                    if ppl_mean < best_val_ppl:
                        print("Achieving better perplexity: {:.8f}\n".format(ppl_mean))
                        best_val_ppl = ppl_mean
                        os.makedirs('./checkpoints/ETM', exist_ok=True)
                        torch.save(
                            model.state_dict(),
                            os.path.join(
                                './checkpoints/ETM',
                                f'{args.dataset}_{args.docs_per_task}shot_best_val_ppl.pth'
                            )
                        )
            print("\n===> Pre-training stage finished in {:.4f}s.".format(time() - t_start))

    else:
        if args.maml_train:
            learner = MetaLearner(args, model)
            ckpt_path = f'./checkpoints/ETM_MAML/{args.dataset}_{args.docs_per_task}shot_best_val_ppl.pth'
            learner.model.load_state_dict(torch.load(ckpt_path, map_location=device))
            print("\n===> Meta-testing stage ===<")
            all_test_ppls = []
            for (support_set, query_set) in test_loader:
                support_set, query_set = support_set.squeeze(0).to(device), query_set.squeeze(0).to(device)
                test_ppl = learner.finetunning(support_set, query_set)
                all_test_ppls.append(test_ppl.item())
            ppl_mean, ppl_std = np.array(all_test_ppls).mean(axis=0), np.array(all_test_ppls).std(axis=0)
            print("The average perplexity on {} test tasks: {:.6f} \u00B1 {:.6f}".format(
                len(test_loader), ppl_mean, ppl_std)
            )
        else:
            ckpt_path = f'./checkpoints/ETM/{args.dataset}_{args.docs_per_task}shot_best_val_ppl.pth'
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            print("\n===> Fine-tuning on test set ===<")
            model.eval()
            all_test_ppls = []
            for (support_set, query_set) in test_loader:
                support_set = support_set.squeeze(0).to(device)
                query_set = query_set.squeeze(0).to(device)

                temp_model = deepcopy(model)
                temp_optim = torch.optim.Adam(
                    temp_model.parameters(),
                    lr=args.update_lr
                )
                temp_model.train()
                for _ in range(args.update_step_test):
                    loss, _, _, _ = temp_model(support_set)
                    temp_optim.zero_grad()
                    loss.backward()
                    temp_optim.step()
                temp_model.eval()
                with torch.no_grad():
                    _, _, _, pred = temp_model(support_set)
                    test_ppl = temp_model.get_ppl(query_set.t(), pred.t())
                    all_test_ppls.append(test_ppl.item())
                del temp_model, temp_optim

            ppl_mean, ppl_std = np.array(all_test_ppls).mean(axis=0), np.array(all_test_ppls).std(axis=0)
            print("The average perplexity on {} test tasks: {:.6f} \u00B1 {:.6f}".format(
                len(test_loader), ppl_mean, ppl_std)
            )


if __name__ == '__main__':
    main()
