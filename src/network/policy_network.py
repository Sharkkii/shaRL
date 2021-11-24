#### Policy Network ####

import numpy as np
import torch
import torch.nn as nn

from .network import BasePolicyNetwork


class DiscretePolicyNetwork(BasePolicyNetwork):

    def __init__(
        self,
        policy_network
    ):
        self.network = policy_network if callable(policy_network) else (lambda state: None)

    def reset(
        self
    ):
        pass

    def setup(
        self
    ):
        pass
    
    def __call__(
        self,
        state
    ):
        return self.network(state)

    def P(
        self,
        state
    ):
        p = self.policy_network(state)
        return p

    def logP(
        self,
        state
    ):
        log_p = torch.log(self.P(state))
        return log_p

class ContinuousPolicyNetwork(BasePolicyNetwork):

    def __init__(
        self,
        policy_network
    ):
        self.network = policy_network if callable(policy_network) else (lambda state, noise: None)

    def reset(
        self
    ):
        pass

    def setup(
        self
    ):
        pass
    
    def __call__(
        self,
        state,
        action
    ):
        return self.network(state, action)

    def P(
        self,
        state,
        action
    ):
        p = self.policy_network(state, action)
        return p

    def logP(
        self,
        state,
        action
    ):
        log_p = torch.log(self.P(state, action))
        return log_p

PolicyNetwork = DiscretePolicyNetwork

class GaussianPolicyNetwork(ContinuousPolicyNetwork):

    class _N(nn.Module):

        def __init__(
            self,
            network_mu,
            network_sigma
        ):
            super().__init__()
            self.network_mu = network_mu
            self.network_sigma = network_sigma

        def forward(
            self,
            state
        ):
            mu = self.network_mu(state)
            log_sigma = self.network_sigma(state)
            sigma = torch.exp(log_sigma)
            return mu, sigma
            
    def __init__(
        self,
        network_mu,
        network_sigma,
        k = 2.0
    ):
        self.network = GaussianPolicyNetwork._N(
            network_mu,
            network_sigma
        )
        self.k = torch.tensor(k)
    
    def __call__(
        self,
        state,
        action
    ):
        return self.P(state, action)

    def P(
        self,
        state,
        action
    ):
        mu, sigma = self.network(state)
        p = GaussianPolicyNetwork._p(action, mu, sigma, k=self.k)
        return p
    
    def logP(
        self,
        state,
        action,
        eps = 1e-8
    ):
        mu, sigma = self.network(state)
        logp = GaussianPolicyNetwork._logp(action, mu, sigma, k=self.k, eps=eps)
        if (torch.any(torch.isclose(logp, torch.tensor(0.0)))):
            assert(False)
        return logp

    def sample(
        self,
        state,
        n_sample = 1,
        requires_grad = False,
        is_deterministic = False
    ):
        # NOTE:
        # <d_action> -> <n_sample, d_action>
        # <n_batch, d_action> -> <n_sample, n_batch, d_action>
        with torch.set_grad_enabled(requires_grad):

            mu, sigma = self.network(state)
            shape = mu.shape

            mu = torch.unsqueeze(mu, dim=0)
            sigma = torch.unsqueeze(sigma, dim=0)
            eps = torch.zeros(n_sample, *shape) if is_deterministic else torch.randn((n_sample, *shape))
            action = mu + eps * sigma
            # NOTE: squashing function (tanh)
            action = self.k * torch.tanh(action)

            if (n_sample == 1):
                action = torch.squeeze(action, dim=0)

        return action

    def _p(
        x,
        mu,
        sigma,
        k = 2.0
    ):
        eps = 1e-8
        margin = 1e-4
        x_squashed = x
        x = x / 2.0
        x = torch.where(x >= 1.0 - margin, x - margin, x)
        x = torch.where(x <= - 1.0 + margin, x + margin, x)
        x_unsquashed = torch.atanh(x)

        p = torch.exp(- (x_unsquashed - mu) ** 2 / (2.0 * sigma ** 2 + eps)) / ((2.0 * np.pi * sigma ** 2 + eps) ** 0.5)
        # NOTE: squashing function (tanh)
        p = p * k / ((k + x_squashed) * (k - x_squashed) + eps)

        # p = torch.prod(p)
        return p

    def _logp(
        x,
        mu,
        sigma,
        k = 2.0,
        eps = 1e-8
    ):
        eps = 1e-4
        eps_in_log = 1e-16
        margin = 1e-4
        x_squashed = x
        x = x / 2.0
        x = torch.where(x >= 1.0 - margin, x - margin, x)
        x = torch.where(x <= - 1.0 + margin, x + margin, x)
        x_unsquashed = torch.atanh(x)
        
        logp = - torch.log(2.0 * np.pi * sigma ** 2 + eps_in_log) / 2.0 - (x_unsquashed - mu) ** 2 / (2.0 * sigma ** 2 + eps)
        # NOTE: squashing function (tanh)
        logp = logp + torch.log(k) - torch.log(k ** 2 - x_squashed ** 2 + eps_in_log)

        # logp = torch.sum(logp)
        return logp
        