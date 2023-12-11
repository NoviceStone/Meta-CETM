import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(act):
    if act == 'tanh':
        act = nn.Tanh()
    elif act == 'relu':
        act = nn.ReLU()
    elif act == 'softplus':
        act = nn.Softplus()
    elif act == 'sigmoid':
        act = nn.Sigmoid()
    elif act == 'leakyrelu':
        act = nn.LeakyReLU()
    elif act == 'elu':
        act = nn.ELU()
    elif act == 'selu':
        act = nn.SELU()
    elif act == 'glu':
        act = nn.GLU()
    else:
        print('Defaulting to tanh activations...')
        act = nn.Tanh()
    return act


class ETM(nn.Module):
    """PyTorch implementation of <<Topic Modeling in Embedding space>> (Dieng et al., TACL 2020)
    Adapted from adjidieng's code: https://github.com/adjidieng/ETM/blob/master/etm.py

    Args
        args: the set of arguments used to characterize the neural topic model.
        device: the physical hardware that the model is trained on.
        word_embeddings: if not None, initialize each word embedding in the vocabulary with pretrained Glove embeddings.
    """

    def __init__(self, args, device, word_embeddings=None):
        super(ETM, self).__init__()
        # define hyper-parameters
        self.device = device
        self.embed_size = args.embed_size
        self.vocab_size = args.vocab_size
        self.num_topics = args.num_topics
        self.num_hiddens = args.num_hiddens
        self.enc_droprate = args.dropout_rate
        self.dropout = nn.Dropout(args.dropout_rate)
        self.realmin = torch.tensor(2.2e-10, dtype=torch.float, device=device)

        if word_embeddings is not None:
            # self.rho = torch.from_numpy(word_embeddings).float().to(device)
            self.rho = nn.Parameter(torch.from_numpy(word_embeddings).float())
        else:
            # learnable word embeddings
            self.rho = nn.Linear(args.embed_size, args.vocab_size, bias=False)

        # latent topic embeddings
        self.alpha = nn.Linear(args.embed_size, args.num_topics, bias=False)

        # deterministic mapping to obtain hidden features
        self.activation = get_activation(args.act)
        self.q_theta = nn.Sequential(
            nn.Linear(args.vocab_size, args.num_hiddens),
            self.activation,
            nn.Linear(args.num_hiddens, args.num_hiddens),
            self.activation,
        )

        # variational encoder to obtain posterior parameters
        self.mu_q_theta = nn.Linear(self.num_hiddens, self.num_topics)
        self.logvar_q_theta = nn.Linear(self.num_hiddens, self.num_topics)

    def reparameterize(self, mu, logvar):
        """Returns a sample from a Gaussian distribution via reparameterization.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)
        else:
            return mu

    @staticmethod
    def kl_normal_normal(mu_theta, logvar_theta):
        """Returns the Kullback-Leibler divergence between a normal distribution and a standard normal distribution.
        """
        kl_div = -0.5 * torch.sum(
            1 + logvar_theta - mu_theta.pow(2) - logvar_theta.exp(), dim=-1
        )
        return kl_div.mean()

    def encode(self, normalized_bows):
        """Returns paramters of the variational distribution for topic proportions θ.
        """
        hidden_feats = self.q_theta(normalized_bows)
        if self.enc_droprate > 0:
            hidden_feats = self.dropout(hidden_feats)
        mu_theta = self.mu_q_theta(hidden_feats)
        logvar_theta = self.logvar_q_theta(hidden_feats)
        sample_theta = self.reparameterize(mu_theta, logvar_theta)
        return mu_theta, logvar_theta, sample_theta

    def get_beta(self):
        """Derives the topic-word matrix by computing the inner product.
        """
        try:
            logit = self.alpha(self.rho.weight)
        except:
            logit = self.alpha(self.rho)
        return torch.softmax(logit, dim=0)

    @staticmethod
    def decode(theta, beta):
        """compute the probability of topic given the document which is equal to θ^T ** B
        """
        res = torch.mm(theta, beta.t())
        almost_zeros = torch.full_like(res, 1e-10)
        results_without_zeros = res.add(almost_zeros)
        return results_without_zeros

    def get_ppl(self, x, x_hat):
        """Returns the evaluation performance of Perplexity.
        """
        x_hat = x_hat / (x_hat.sum(0) + self.realmin)
        ppl = -1.0 / x.sum() * x * torch.log(x_hat + self.realmin)
        return torch.exp(ppl.sum())

    def forward(self, bows, fast_weights=None):
        """Forward pass: compute the kl loss and data likelihood.
        """
        denominator = torch.where(
            bows.sum(dim=1, keepdims=True) > 0,
            bows.sum(dim=1, keepdims=True),
            torch.tensor([1.]).to(self.device)
        )
        normalized_bows = bows / denominator

        if fast_weights is None:
            sample_theta, mu_theta, logvar_theta = self.encode(normalized_bows)
            theta = torch.softmax(sample_theta, dim=-1)
            beta = self.get_beta()
            results_without_zeros = self.decode(theta, beta)
            predictions = torch.log(results_without_zeros)
        else:
            fast_weights = {name: fast_weights[i] for i, (name, _) in enumerate(self.named_parameters())}
            h = F.relu(F.linear(
                normalized_bows, fast_weights['q_theta.0.weight'], fast_weights['q_theta.0.bias']
            ))
            h = F.relu(F.linear(
                h, fast_weights['q_theta.2.weight'], fast_weights['q_theta.2.bias']
            ))
            h = F.dropout(
                h, self.enc_droprate, self.training, False
            )
            mu_theta = F.linear(
                h, fast_weights['mu_q_theta.weight'], fast_weights['mu_q_theta.bias']
            )
            logvar_theta = F.linear(
                h, fast_weights['logvar_q_theta.weight'], fast_weights['logvar_q_theta.bias']
            )
            sample_theta = self.reparameterize(mu_theta, logvar_theta)
            theta = torch.softmax(sample_theta, dim=-1)
            if isinstance(self.rho, nn.Linear):
                beta = torch.softmax(torch.mm(
                    fast_weights['rho.weight'], fast_weights['alpha.weight'].t()
                ), dim=0)
            else:
                beta = torch.softmax(torch.mm(
                    fast_weights['rho'], fast_weights['alpha.weight'].t()
                ), dim=0)
            res = torch.mm(theta, beta.t())
            almost_zeros = torch.full_like(res, 1e-10)
            results_without_zeros = res.add(almost_zeros)
            predictions = torch.log(results_without_zeros)

        kl_loss = self.kl_normal_normal(mu_theta, logvar_theta)
        neg_log_likelihood = -(predictions * bows).sum(1).mean()
        neg_elbo = neg_log_likelihood + kl_loss
        return neg_elbo, neg_log_likelihood, kl_loss, results_without_zeros
