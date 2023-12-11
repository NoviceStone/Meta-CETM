import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import AttentiveStatistiC, ThetaPosteriorNet, GraphEncoder, InnerProductDecoder


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


class MetaContextETM(nn.Module):
    """Simple implementation of the <<Sawtooth Factorial Topic Embeddings Guided Gamma Belief Network>>

    Args
        args: the set of arguments used to characterize the hierarchical neural topic model.
        device: the physical hardware that the model is trained on.
        word_embeddings: if not None, initialize each word embedding in the vocabulary with pretrained Glove embeddings.
    """

    def __init__(self, args, device, word_embeddings):
        super(MetaContextETM, self).__init__()
        self.device = device
        self.embed_size = args.embed_size
        self.vocab_size = args.vocab_size
        self.num_topics = args.num_topics
        self.num_hiddens = args.num_hiddens

        self.z_dim = args.z_dim
        self.fix_pi = args.fix_pi
        self.em_iterations = args.em_iterations
        self.train_mc_sample_size = args.train_mc_sample_size
        self.real_min = torch.tensor(2.2e-10, dtype=torch.float, device=device)

        if word_embeddings is not None:
            self.rho = torch.from_numpy(word_embeddings).float().to(device)
        else:
            # learnable word embeddings
            self.rho = nn.Parameter(
                torch.empty(args.vocab_size, args.embed_size).normal_(std=0.02))

        # variational bows encoder: to obtain posteriors of latent variables (c and theta)
        self.activation = get_activation(args.act)
        self.shared_x_encoder = nn.Sequential(
            nn.Linear(args.vocab_size, args.num_hiddens),
            nn.BatchNorm1d(args.num_hiddens),
            self.activation,
            nn.Linear(args.num_hiddens, args.num_hiddens),
            nn.BatchNorm1d(args.num_hiddens),
            self.activation
        )
        self.q_c_given_x_net = AttentiveStatistiC(
            hid_dim=args.num_hiddens,
            c_dim=args.num_topics,
            activation=self.activation
        )
        self.q_theta_given_x_c_net = ThetaPosteriorNet(
            num_layers=2,
            hid_dim=args.num_hiddens,
            c_dim=args.num_topics,
            activation=self.activation
        )

        # variational graph encoder: to obtain posteriors of latent word embeddings (z)
        self.graph_encoder = GraphEncoder(
            in_channels=args.embed_size,
            hidden_channels=args.graph_h_dim,
            out_channels=args.z_dim,
            activation=self.activation,
        )

        # graph decoder: to reconstruct dependency graph
        self.graph_decoder = InnerProductDecoder(
            activation=get_activation('sigmoid'),
            droprate=0.
        )

        self.register_buffer('log_norm_constant', torch.tensor(-0.5 * math.log(2 * math.pi)))
        self.register_buffer('uniform_pi', torch.ones(self.num_topics)/self.num_topics)

    def reparameterize(self, mean, logvar):
        """Returns a sample from a Gaussian distribution via reparameterization.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mean)
        else:
            return mean

    @staticmethod
    def kl_normal_normal(q_mean, q_logvar, p_mean=None, p_logvar=None):
        """Returns the Kullback-Leibler divergence between two Gaussian distributions.
        """
        if p_mean is not None and p_logvar is not None:
            kl_div = 0.5 * torch.sum(
                ((q_mean - p_mean).pow(2) + q_logvar.exp()) / (p_logvar.exp() + 1e-8) - 1 - q_logvar + p_logvar,
                dim=-1
            )
        else:
            kl_div = 0.5 * torch.sum(q_mean.pow(2) + q_logvar.exp() - 1 - q_logvar, dim=-1)
        return kl_div.mean()

    def gaussian_log_prob(self, x, mean, logvar=None):
        """Returns the log-normal probability density of the datapoint x.
        """
        if logvar is None:
            logvar = torch.zeros_like(mean)
        log_px = -0.5 * (logvar + (x - mean).pow(2) / logvar.exp())
        log_px = log_px + self.log_norm_constant
        return log_px.sum(dim=-1)

    def EM_step(self, Z, psi):
        """Returns the updated prior parameters by performing E-step and M-step.
        """
        sample_size = Z.size(0)
        pi, mu, log_sigma2 = psi

        weighted_log_probs = self.gaussian_log_prob(
            Z[:, None, :].repeat(1, self.num_topics, 1),           # num_nodes, num_topics, latent_size
            mu[None, :, :].repeat(sample_size, 1, 1),              # num_nodes, num_topics, latent_size
            log_sigma2[None, :, :].repeat(sample_size, 1, 1)       # num_nodes, num_topics, latent_size
        ) + torch.log(pi[None, :].repeat(sample_size, 1))          # num_nodes, num_topics

        """E-Step"""
        responsibilities = torch.exp(                              # num_nodes, num_topics
            weighted_log_probs - torch.logsumexp(weighted_log_probs, dim=-1, keepdim=True)
        )

        """M-Step"""
        # update pi
        N = torch.sum(responsibilities, dim=0)                     # num_topics
        if not self.fix_pi:
            pi_new = N / N.sum()
        else:
            pi_new = pi

        # update mu
        denominator = N[:, None].repeat(1, self.z_dim)             # num_topics, latent_size
        mu_new = torch.sum(                                        # num_topics, latent_size
            responsibilities[:, :, None].repeat(                   # num_nodes, num_topics, latent_size
                1, 1, self.z_dim) * Z[:, None, :].repeat(1, self.num_topics, 1),
            dim=0
        ) / denominator

        # update sigma2
        sigma2_new = torch.sum(
            responsibilities[:, :, None].repeat(1, 1, self.z_dim) * (Z[:, None, :].repeat(
                1, self.num_topics, 1) - mu_new[None, :, :].repeat(sample_size, 1, 1)).pow(2),
            dim=0
        ) / denominator
        return pi_new, mu_new, torch.log(sigma2_new + 1e-8)

    def get_z_mixture_prior(self, Z):
        """Returns the gaussian mixture prior parameters for latent (word) embeddings.
        """
        initial_pi = self.uniform_pi
        select_ids = torch.from_numpy(
            np.random.choice(Z.size(0), self.num_topics, replace=False)
        ).to(self.device)
        initial_mu = torch.index_select(Z, dim=0, index=select_ids)
        initial_log_sigma2 = torch.zeros_like(initial_mu)

        psi = (initial_pi, initial_mu, initial_log_sigma2)
        for _ in range(self.em_iterations):
            psi = self.EM_step(Z=Z, psi=psi)
        psi = (param.detach() for param in psi)
        return psi

    def get_z_posterior(self, E, adj):
        """Returns the gaussian posteriors for the latent codes of word embeddings.
        """
        edge_index = torch.nonzero(adj)
        edge_weight = adj[edge_index[:, 0], edge_index[:, 1]]
        z_mean, z_logvar = self.graph_encoder(E, edge_index.t(), edge_weight)
        z_sample = self.reparameterize(
            mean=z_mean,
            logvar=z_logvar,
        )
        return z_mean, z_logvar, z_sample

    def get_c_posterior(self, H):
        """Returns the gaussian posteriors for latent variables c (prior of topic proportions).
        """
        c_mean, c_logvar, _, _ = self.q_c_given_x_net(H)
        c_sample = self.reparameterize(
            mean=c_mean,
            logvar=c_logvar,
        )
        return c_mean, c_logvar, c_sample

    def get_theta_posterior(self, H, c):
        """Returns the logistic-normal posteriors for latent variables θ (topic proportions).
        """
        theta_mean, theta_logvar = self.q_theta_given_x_c_net(H, c)
        theta_sample = self.reparameterize(
            mean=theta_mean,
            logvar=theta_logvar,
        )
        return theta_mean, theta_logvar, theta_sample

    def get_ppl(self, x, x_hat):
        """Returns perplexity as a measure of the performance.
        """
        x_hat = x_hat / (x_hat.sum(0) + self.real_min)
        ppl = -1.0 / x.sum() * x * torch.log(x_hat + self.real_min)
        return torch.exp(ppl.sum())
        # return ppl.sum(0)

    def forward(self, X, adj, ind):
        """Forward pass: inference and generation, computes the data likelihood.
        """
        # ========================= Variational Inference for Latent Variables =========================
        # infer c and θ
        X_norm = X / X.sum(1, keepdim=True)
        hidden_feats = self.shared_x_encoder(X_norm)
        c_mean, c_logvar, c_sample = self.get_c_posterior(hidden_feats)
        theta_mean, theta_logvar, theta_sample = self.get_theta_posterior(hidden_feats, c_sample)
        theta = torch.softmax(theta_sample, dim=-1)

        c_kl = self.kl_normal_normal(
            q_mean=c_mean,
            q_logvar=c_logvar,
        )
        theta_kl = self.kl_normal_normal(
            q_mean=theta_mean,
            q_logvar=theta_logvar,
            p_mean=c_sample.expand_as(theta_mean),
            p_logvar=torch.zeros_like(theta_logvar),
        )

        # infer z
        num_nodes = ind.size(0)
        node_features = torch.mm(ind, self.rho)
        z_mean, z_logvar, z_sample = self.get_z_posterior(node_features, adj)

        # get parameters of the Gaussian mixture prior by EM algorithm
        psi = self.get_z_mixture_prior(z_sample)
        gmm_pi, gmm_mu, gmm_logvar = psi

        # kl_div = E_qz[log(qz/pz)]
        log_qz = torch.sum(-0.5 * (1 + z_logvar) + self.log_norm_constant, dim=-1)
        log_likelihoods = self.gaussian_log_prob(
            z_sample[:, None, :].repeat(1, self.num_topics, 1),
            gmm_mu[None, :, :].repeat(num_nodes, 1, 1),
            gmm_logvar[None, :, :].repeat(num_nodes, 1, 1)
        ) + torch.log(gmm_pi[None, :].repeat(num_nodes, 1))
        log_pz = torch.logsumexp(log_likelihoods, dim=-1)
        z_kl = torch.mean(log_qz - log_pz)

        # ======================== Generation for the BoWs and Dependency Graph ========================
        # reconstruct graph
        pos_weight = float(adj.size(0) ** 2 - adj.sum()) / adj.sum()
        weight_mask = (adj == 1)
        weight_tensor = torch.ones(weight_mask.size()).to(self.device)
        weight_tensor[weight_mask] = pos_weight

        adj_rec = self.graph_decoder(z_sample)
        graph_likelihood = F.binary_cross_entropy(
            input=adj_rec,
            target=adj,
            weight=weight_tensor,
            reduction='mean'
        )

        # reconstruct bow
        norm_log_likelihoods = F.normalize(log_likelihoods, dim=1)
        t_over_w = torch.softmax(norm_log_likelihoods, dim=0)
        beta = torch.mm(ind.t(), t_over_w)
        X_rec = torch.mm(theta, beta.t())
        bow_likelihood = torch.mean((-torch.log(X_rec + 1e-8) * X).sum(1))

        elbo = bow_likelihood + 0.1 * (c_kl + theta_kl) + (graph_likelihood + z_kl)
        return elbo, bow_likelihood, graph_likelihood, c_kl, theta_kl, z_kl, ind.t()

    def predict(self, X, adj, ind, beta=None):
        """Pre: returns the reconstructed observations.
        """
        denorm = torch.where(
            X.sum(dim=1, keepdims=True) > 0,
            X.sum(dim=1, keepdims=True),
            torch.tensor([1.]).to(self.device)
        )
        X_norm = X / denorm
        hidden_feats = self.shared_x_encoder(X_norm)
        c_mean, c_logvar, c_sample = self.get_c_posterior(hidden_feats)
        theta_mean, theta_logvar, theta_sample = self.get_theta_posterior(hidden_feats, c_sample)
        theta = torch.softmax(theta_sample, dim=-1)

        if beta is None:
            num_nodes = ind.size(0)
            node_features = torch.mm(ind, self.rho)
            z_mean, z_logvar, z_sample = self.get_z_posterior(node_features, adj)

            psi = self.get_z_mixture_prior(z_sample)
            gmm_pi, gmm_mu, gmm_logvar = psi
            log_likelihoods = self.gaussian_log_prob(
                z_sample[:, None, :].repeat(1, self.num_topics, 1),
                gmm_mu[None, :, :].repeat(num_nodes, 1, 1),
                gmm_logvar[None, :, :].repeat(num_nodes, 1, 1)
            ) + torch.log(gmm_pi[None, :].repeat(num_nodes, 1))

            norm_log_likelihoods = F.normalize(log_likelihoods, dim=1)
            t_over_w = torch.softmax(norm_log_likelihoods, dim=0)
            beta = torch.mm(ind.t(), t_over_w)

        X_rec = torch.mm(theta, beta.t())
        return X_rec, beta
