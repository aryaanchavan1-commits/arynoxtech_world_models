import torch
import torch.nn.functional as F

def reconstruction_loss(pred_obs, target_obs):
    """
    MSE loss for observation reconstruction.
    """
    return F.mse_loss(pred_obs, target_obs)

def reward_loss(pred_reward, target_reward):
    """
    MSE loss for reward prediction.
    """
    return F.mse_loss(pred_reward, target_reward)

def kl_divergence_loss(posterior_dist, prior_dist):
    """
    KL divergence between posterior and prior.
    """
    return torch.distributions.kl_divergence(posterior_dist, prior_dist).mean()

def value_loss(pred_value, target_value):
    """
    MSE loss for value prediction.
    """
    return F.mse_loss(pred_value, target_value)