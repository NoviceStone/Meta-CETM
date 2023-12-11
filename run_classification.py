import argparse
import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data

from datasets import PlainClsDataset, MetaClsDataset
from models.clsnet import ClassificationNet
from models.maml import MetaClsLearner


parser = argparse.ArgumentParser(description='run experiment for few-shot classification.')

# data-related arguments
parser.add_argument('--seed', type=int, default=2023, help='random seed.')
parser.add_argument('--dataset', type=str, default='', help='name of corpus.')
parser.add_argument('--data_path', type=str, default='', help='path to load dataset.')
parser.add_argument('--batch_size', type=int, default=200, help='input batch size for training.')

# optimization-related arguments
parser.add_argument('--strategy', type=str, default='proto', help='could be one of [maml, proto, ft].')
parser.add_argument('--arch', type=str, default='conv', help='could be one of [conv, mlp].')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train.')
parser.add_argument('--optimizer', type=str, default='adam', help='choice of optimizer.')
parser.add_argument('--lr', type=float, default=0.005, help='learning rate.')
parser.add_argument('--grad_clip', type=float, default=20.0, help='gradient clipping.')
parser.add_argument('--weight_decay', type=float, default=1.2e-6, help='some l2 regularization.')

# few-shot setting arguments
parser.add_argument('--meta_batch_size', type=int, help='number of tasks processed at each update', default=5)
parser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=3e-4)
parser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=3e-3)
parser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
parser.add_argument('--update_step_test', type=int, help='update steps for fine-tuning', default=10)
parser.add_argument('--num_ways', type=int, default=5, help='number of classes for each N-way-K-shot task.')
parser.add_argument('--num_shots', type=int, default=10, help='number of support samples in each N-way-K-shot task.')
parser.add_argument('--num_queries', type=int, default=15, help='number of query samples in each N-way-K-shot task.')
args = parser.parse_args()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # data loading pipeline
    test_set = MetaClsDataset(
        dataset_name=args.dataset,
        data_path=args.data_path,
        mode='test',
        num_ways=args.num_ways,
        num_shots=args.num_shots
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=1, shuffle=False, num_workers=2
    )

    # define the directory to save models
    if args.arch == 'conv':
        save_dir = './checkpoints/classification/CNN'
    elif args.arch == 'mlp':
        save_dir = './checkpoints/classification/MLP'
    else:
        save_dir = f'./checkpoints/classification/{args.arch}'
    os.makedirs(save_dir, exist_ok=True)

    # three typical few-shot learning algorithms
    if args.strategy == 'maml':
        train_set = MetaClsDataset(
            dataset_name=args.dataset,
            data_path=args.data_path,
            mode='train',
            num_ways=args.num_ways,
            num_shots=args.num_shots
        )
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.meta_batch_size, shuffle=True, num_workers=4
        )
        model = ClassificationNet(
            input_dim=len(train_set.vocab),
            num_classes=args.num_ways,
            arch=args.arch
        )
        model.to(device)
        print('\nModel : {}'.format(model))
        tmp = filter(lambda x: x.requires_grad, model.parameters())
        num = sum(map(lambda x: np.prod(x.shape), tmp))
        print('Total trainable tensors:', num)
        learner = MetaClsLearner(args, model, device)
        print('\nApplied few-shot algorithm: model-agnostic meta-learning')
        print('\n===> Meta-training stage ===<')
        for epoch in range(8):
            print('\n===> Epoch: {} <==='.format(epoch))
            train_loss = []
            train_acc = []
            model.train()
            for idx, (sup_batches, qry_batches) in enumerate(tqdm(train_loader)):
                sup_batches, qry_batches = sup_batches.to(device), qry_batches.to(device)
                sup_batches = sup_batches.view(args.meta_batch_size, args.num_ways * args.num_shots, -1)
                qry_batches = qry_batches.view(args.meta_batch_size, args.num_ways * args.num_queries, -1)
                loss_qry, acc_qry = learner(sup_batches, qry_batches)
                train_loss.append(loss_qry)
                train_acc.append(acc_qry)
                if (idx + 1) % 100 == 0:
                    print('Avg Train Loss: {}, Avg Train Acc: {}'.format(np.mean(train_loss), np.mean(train_acc)))
            # save model weights once per epoch
            torch.save(
                learner.model.state_dict(),
                os.path.join(
                    save_dir,
                    f'{args.dataset}_{args.num_shots}shot_{args.strategy}_epoch{epoch+1}_model.pth'
                )
            )

        print('\n===> Meta-testing stage ===<')
        ckpt_path = os.path.join(
            save_dir,
            f'{args.dataset}_{args.num_shots}shot_{args.strategy}_epoch8_model.pth'
        )
        learner.model.load_state_dict(torch.load(ckpt_path, map_location=device))
        accs_all_test = []
        for (sup_batch, qry_batch) in tqdm(test_loader):
            sup_batch = sup_batch.squeeze(0).to(device)
            qry_batch = qry_batch.squeeze(0).to(device)
            sup_batch = sup_batch.view(args.num_ways * args.num_shots, -1)
            qry_batch = qry_batch.view(args.num_ways * args.num_queries, -1)
            acc = learner.finetunning(sup_batch, qry_batch)
            accs_all_test.append(acc.item())
        mu_acc, std_acc = np.array(accs_all_test).mean(), np.array(accs_all_test).std()
        print("The average accuracy on {} test tasks: {:.6f} \u00B1 {:.6f}".format(
            len(test_loader), mu_acc, std_acc))

    elif args.strategy == 'proto':
        train_set = MetaClsDataset(
            dataset_name=args.dataset,
            data_path=args.data_path,
            mode='train',
            num_ways=args.num_ways,
            num_shots=args.num_shots
        )
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=1, shuffle=True, num_workers=0
        )
        model = ClassificationNet(
            input_dim=len(train_set.vocab),
            num_classes=args.num_ways,
            arch=args.arch
        )
        model.to(device)
        print('\nModel : {}'.format(model))
        tmp = filter(lambda x: x.requires_grad, model.parameters())
        num = sum(map(lambda x: np.prod(x.shape), tmp))
        print('Total trainable tensors:', num)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        print('\nApplied few-shot algorithm: prototypical network')
        print('\n===> Meta-training stage ===<')
        best_train_acc = 0.
        for epoch in range(5):
            print('\n===> Epoch: {} <==='.format(epoch))
            train_loss = []
            train_acc = []
            model.train()
            qry_labels = torch.arange(args.num_ways).unsqueeze(1).repeat(1, args.num_queries).view(-1).to(device)
            for idx, (sup_batch, qry_batch) in enumerate(tqdm(train_loader)):
                sup_batch = sup_batch.squeeze(0).to(device)
                qry_batch = qry_batch.squeeze(0).to(device)
                sup_feats, _ = model(sup_batch.view(args.num_ways * args.num_shots, -1))
                sup_proto = sup_feats.view(args.num_ways, args.num_shots, -1).mean(1)
                qry_feats, _ = model(qry_batch.view(args.num_ways * args.num_queries, -1))
                qry_dists = -(qry_feats.unsqueeze(1) - sup_proto.unsqueeze(0)).pow(2).sum(-1)
                loss = criterion(qry_dists, qry_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
                num_correct = torch.eq(qry_dists.argmax(-1), qry_labels).float().sum()
                acc = num_correct / (args.num_ways * args.num_queries)
                train_acc.append(acc.item())
                if (idx + 1) % 500 == 0:
                    if np.mean(train_acc) > best_train_acc:
                        best_train_acc = np.mean(train_acc)
                        torch.save(
                            model.state_dict(),
                            os.path.join(
                                save_dir,
                                f'{args.dataset}_{args.num_shots}shot_{args.strategy}_best_model.pth'
                            )
                        )
                    print('Avg Train Loss: {}, Avg Train Acc: {}'.format(np.mean(train_loss), np.mean(train_acc)))
                    train_acc.clear()

        print('\n===> Meta-testing stage ===<')
        ckpt_path = os.path.join(
            save_dir,
            f'{args.dataset}_{args.num_shots}shot_{args.strategy}_best_model.pth'
        )
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        qry_labels = torch.arange(args.num_ways).unsqueeze(1).repeat(1, args.num_queries).view(-1).to(device)
        model.eval()
        accs_all_test = []
        for (sup_batch, qry_batch) in tqdm(test_loader):
            sup_batch = sup_batch.squeeze(0).to(device)
            qry_batch = qry_batch.squeeze(0).to(device)
            with torch.no_grad():
                sup_feats, _ = model(sup_batch.view(args.num_ways * args.num_shots, -1))
                qry_feats, _ = model(qry_batch.view(args.num_ways * args.num_queries, -1))
            sup_proto = sup_feats.view(args.num_ways, args.num_shots, -1).mean(1)
            qry_dists = (qry_feats.unsqueeze(1) - sup_proto.unsqueeze(0)).pow(2).sum(-1)
            num_correct = torch.eq(qry_dists.argmin(-1), qry_labels).float().sum()
            acc = num_correct / (args.num_ways * args.num_queries)
            accs_all_test.append(acc.item())
        mu_acc, std_acc = np.array(accs_all_test).mean(), np.array(accs_all_test).std()
        print("The average accuracy on {} test tasks: {:.6f} \u00B1 {:.6f}".format(
            len(test_loader), mu_acc, std_acc))

    elif args.strategy == 'ft':
        train_set = PlainClsDataset(args.dataset, args.data_path, 'train')
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4
        )
        model = ClassificationNet(
            input_dim=len(train_set.vocab),
            num_classes=len(train_set.class_id),
            arch=args.arch
        )
        model.to(device)
        print('\nModel : {}'.format(model))
        tmp = filter(lambda x: x.requires_grad, model.parameters())
        num = sum(map(lambda x: np.prod(x.shape), tmp))
        print('Total trainable tensors:', num)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            gamma=0.2,
            step_size=30
        )
        criterion = nn.CrossEntropyLoss()
        print('\nApplied few-shot algorithm: baseline')
        print('\n===> Pre-training stage ===<')
        best_train_acc = 0.
        for epoch in tqdm(range(args.epochs)):
            train_loss = []
            train_acc = []
            model.train()
            for idx, (batch_data, batch_labels) in enumerate(train_loader):
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                _, output = model(batch_data)
                loss = criterion(output, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
                num_correct = torch.eq(output.argmax(-1), batch_labels).float().sum()
                acc = num_correct / batch_labels.shape[0]
                train_acc.append(acc.item())
            print('Epoch: {}, Avg Train Loss: {}, Avg Train Acc: {}'.format(
                epoch, np.mean(train_loss), np.mean(train_acc)))
            if np.mean(train_acc) > best_train_acc:
                best_train_acc = np.mean(train_acc)
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        save_dir,
                        f'{args.dataset}_{args.num_shots}shot_{args.strategy}_best_model.pth'
                    )
                )
            scheduler.step()

        print("\n===> Fine-tuning on meta-test set ===<")
        ckpt_path = os.path.join(
            save_dir,
            f'{args.dataset}_{args.num_shots}shot_{args.strategy}_best_model.pth'
        )
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        sup_labels = torch.arange(args.num_ways).unsqueeze(1).repeat(1, args.num_shots).view(-1).to(device)
        qry_labels = torch.arange(args.num_ways).unsqueeze(1).repeat(1, args.num_queries).view(-1).to(device)
        model.eval()
        accs_all_test = []
        for (sup_batch, qry_batch) in tqdm(test_loader):
            sup_batch = sup_batch.squeeze(0).to(device)
            qry_batch = qry_batch.squeeze(0).to(device)
            with torch.no_grad():
                sup_feats, _ = model(sup_batch.view(args.num_ways * args.num_shots, -1))
                qry_feats, _ = model(qry_batch.view(args.num_ways * args.num_queries, -1))
            # temp_clf = nn.Linear(128, args.num_ways).to(device)  # for MLP arch
            temp_clf = nn.Linear(2984, args.num_ways).to(device)  # for CNN arch
            temp_optim = torch.optim.Adam(temp_clf.parameters())
            for _ in range(100):
                sup_outputs = temp_clf(sup_feats)
                loss = criterion(sup_outputs, sup_labels)
                temp_optim.zero_grad()
                loss.backward()
                temp_optim.step()
            qry_outputs = temp_clf(qry_feats)
            num_correct = torch.eq(qry_outputs.argmax(-1), qry_labels).float().sum()
            acc = num_correct / (args.num_ways * args.num_queries)
            accs_all_test.append(acc.item())
            del temp_optim, temp_clf
        mu_acc, std_acc = np.array(accs_all_test).mean(), np.array(accs_all_test).std()
        print("The average accuracy on {} test tasks: {:.6f} \u00B1 {:.6f}".format(
            len(test_loader), mu_acc, std_acc))

    else:
        raise NotImplementedError(f"The '{args.strategy}' strategy has not been implemented!")


if __name__ == '__main__':
    main()
