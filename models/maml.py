import torch
from torch import nn
from copy import deepcopy


class MetaLearner(nn.Module):
    """Meta Learner: implement the functionality described in the paper
    <<Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks>> (Finn et al., ICML 2017).
    """

    def __init__(self, args, base_model):
        super(MetaLearner, self).__init__()
        self.meta_lr = args.meta_lr
        self.update_lr = args.update_lr
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test

        self.model = base_model
        self.meta_optim = torch.optim.Adam(
            self.model.parameters(), lr=self.meta_lr
        )
        self.grad_clip = args.grad_clip

    def forward(self, support_batches, query_batches):
        n_tasks, task_size, vocab_size = support_batches.size()

        # losses_q[k] is the loss on query set of updated step k
        losses_q = [0 for _ in range(self.update_step + 1)]

        for n in range(n_tasks):
            # run the n-th task and compute loss for k=0
            nelbo_s, _, _, _ = self.model(support_batches[n], fast_weights=None)
            grads = torch.autograd.grad(outputs=nelbo_s, inputs=self.model.parameters())
            fast_weights = list(map(
                lambda x: x[1] - self.update_lr * x[0], zip(grads, self.model.parameters())
            ))

            # this is the loss before the first update
            with torch.no_grad():
                nelbo_q, _, _, _ = self.model(query_batches[n], fast_weights=None)
                losses_q[0] += nelbo_q

            # this is the loss after the first update
            with torch.no_grad():
                nelbo_q, _, _, _ = self.model(query_batches[n], fast_weights=fast_weights)
                losses_q[1] += nelbo_q

            for k in range(1, self.update_step):
                # run the n-th task and compute loss for k=1,...,K-1
                nelbo_s, _, _, _ = self.model(support_batches[n], fast_weights=fast_weights)
                grads = torch.autograd.grad(outputs=nelbo_s, inputs=fast_weights)
                fast_weights = list(map(
                    lambda x: x[1] - self.update_lr * x[0], zip(grads, fast_weights)
                ))

                nelbo_q, _, _, _ = self.model(query_batches[n], fast_weights=fast_weights)
                losses_q[k + 1] += nelbo_q

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / n_tasks

        # optimize theta parameters
        # print('meta update')
        self.meta_optim.zero_grad()
        loss_q.backward()
        for param in list(self.model.parameters()):
            nn.utils.clip_grad_norm_(param, self.grad_clip)
        self.meta_optim.step()
        return loss_q.item()

    def finetunning(self, support_data, query_data):
        assert len(support_data.shape) == 2

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we fine-tunning on the copied model instead of self.net
        temp_model = deepcopy(self.model)

        # 1. run the new task and compute loss for k=0,...,K
        nelbo_s, _, _, _ = temp_model(support_data, fast_weights=None)
        grads = torch.autograd.grad(outputs=nelbo_s, inputs=temp_model.parameters())
        fast_weights = list(map(
            lambda x: x[1] - self.update_lr * x[0], zip(grads, temp_model.parameters())
        ))

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            nelbo_s, _, _, _ = temp_model(support_data, fast_weights=fast_weights)
            grads = torch.autograd.grad(outputs=nelbo_s, inputs=fast_weights)
            fast_weights = list(map(
                lambda x: x[1] - self.update_lr * x[0], zip(grads, fast_weights)
            ))

        temp_model.eval()
        with torch.no_grad():
            _, _, _, pred = temp_model(support_data, fast_weights=fast_weights)
            ppl_test = temp_model.get_ppl(query_data.t(), pred.t())
        del temp_model

        return ppl_test


class MetaClsLearner(nn.Module):
    """Meta Learner: implement the functionality described in the paper
    << Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks >> (Finn et al., ICML 2017).
    """

    def __init__(self, args, base_model, device):
        super(MetaClsLearner, self).__init__()
        self.device = device
        self.meta_lr = args.meta_lr
        self.update_lr = args.update_lr
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test

        self.model = base_model
        self.meta_optim = torch.optim.Adam(
            self.model.parameters(), lr=self.meta_lr
        )
        self.criterion = nn.CrossEntropyLoss()

        self.sup_labels = torch.arange(args.num_ways).unsqueeze(1).repeat(
            1, args.num_shots).view(-1).to(device)
        self.qry_labels = torch.arange(args.num_ways).unsqueeze(1).repeat(
            1, args.num_queries).view(-1).to(device)

    def forward(self, support_batches, query_batches):
        n_tasks, task_size, vocab_size = support_batches.size()

        # losses_q[k] is the loss on query set of updated step k
        losses_q = [0 for _ in range(self.update_step + 1)]
        acc_q = 0.

        for n in range(n_tasks):
            # run the n-th task and compute loss for k=0
            _, output = self.model(support_batches[n], fast_weights=None)
            sup_loss = self.criterion(output, self.sup_labels)
            grads = torch.autograd.grad(outputs=sup_loss, inputs=self.model.parameters())
            fast_weights = list(map(
                lambda x: x[1] - self.update_lr * x[0], zip(grads, self.model.parameters())
            ))

            # this is the loss before the first update
            with torch.no_grad():
                _, output_q = self.model(query_batches[n], fast_weights=None)
                qry_loss = self.criterion(output_q, self.qry_labels)
                losses_q[0] += qry_loss

            # this is the loss after the first update
            with torch.no_grad():
                _, output_q = self.model(query_batches[n], fast_weights=fast_weights)
                qry_loss = self.criterion(output_q, self.qry_labels)
                losses_q[1] += qry_loss

            for k in range(1, self.update_step):
                # run the n-th task and compute loss for k=1,...,K-1
                _, output = self.model(support_batches[n], fast_weights=fast_weights)
                sup_loss = self.criterion(output, self.sup_labels)
                grads = torch.autograd.grad(outputs=sup_loss, inputs=fast_weights)
                fast_weights = list(map(
                    lambda x: x[1] - self.update_lr * x[0], zip(grads, fast_weights)
                ))

                _, output_q = self.model(query_batches[n], fast_weights=fast_weights)
                qry_loss = self.criterion(output_q, self.qry_labels)
                losses_q[k + 1] += qry_loss

            # this is the accuracy after K update step
            correct = torch.eq(output_q.argmax(-1), self.qry_labels).float().sum()
            acc_q += correct / output_q.shape[0]

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / n_tasks
        acc_q = acc_q / n_tasks

        # optimize theta parameters
        # print('meta update')
        self.meta_optim.zero_grad()
        loss_q.backward()
        # for param in list(self.model.parameters()):
        #     nn.utils.clip_grad_norm_(param, self.grad_clip)
        self.meta_optim.step()
        return loss_q.item(), acc_q.item()

    def finetunning(self, support_data, query_data):
        assert len(support_data.shape) == 2

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we fine-tunning on the copied model instead of self.net
        temp_model = deepcopy(self.model)

        # 1. run the new task and compute loss for k=0,...,K
        _, output = temp_model(support_data, fast_weights=None)
        sup_loss = self.criterion(output, self.sup_labels)
        grads = torch.autograd.grad(outputs=sup_loss, inputs=temp_model.parameters())
        fast_weights = list(map(
            lambda x: x[1] - self.update_lr * x[0], zip(grads, temp_model.parameters())
        ))

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            _, output = temp_model(support_data, fast_weights=fast_weights)
            sup_loss = self.criterion(output, self.sup_labels)
            grads = torch.autograd.grad(outputs=sup_loss, inputs=fast_weights)
            fast_weights = list(map(
                lambda x: x[1] - self.update_lr * x[0], zip(grads, fast_weights)
            ))

        temp_model.eval()
        with torch.no_grad():
            _, output = temp_model(query_data, fast_weights=fast_weights)
            acc_test = torch.eq(output.argmax(-1), self.qry_labels).float().sum() / query_data.size(0)
        del temp_model

        return acc_test
