"""
Loss Functions
==============
Loss functions for VLA training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional


class FlowMatchingLoss(nn.Module):
    """
    Flow Matching Loss
    
    Computes the conditional flow matching objective:
    L = E[||v_θ(x_t, t) - (x_1 - x_0)||²]
    
    Where:
    - x_0: Source distribution (noise)
    - x_1: Target distribution (data)
    - x_t: Interpolated state at time t
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = config or {}
    
    def forward(
        self, 
        predicted_velocity: torch.Tensor, 
        target_velocity: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute flow matching loss
        
        Args:
            predicted_velocity: Predicted velocity field
            target_velocity: Target velocity (x_1 - x_0)
            
        Returns:
            loss: Scalar loss value
        """
        loss = F.mse_loss(predicted_velocity, target_velocity)
        return loss
    
    def compute(
        self, 
        model: nn.Module, 
        x: torch.Tensor, 
        action_0: torch.Tensor, 
        action_1: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss with model
        
        Args:
            model: Velocity prediction model
            x: Condition features
            action_0: Source actions (noise)
            action_1: Target actions
            
        Returns:
            loss: Scalar loss value
        """
        B = x.shape[0]
        device = x.device
        
        # Sample random time
        t = torch.rand(B, device=device)
        
        # Interpolate
        action_t = t.unsqueeze(-1).unsqueeze(-1) * action_1 + \
                   (1 - t.unsqueeze(-1).unsqueeze(-1)) * action_0
        
        # Target velocity
        target_velocity = action_1 - action_0
        
        # Predict velocity
        predicted_velocity = model(x, t, action_t)
        
        # Compute loss
        loss = self.forward(predicted_velocity, target_velocity)
        
        return loss


class DiffusionLoss(nn.Module):
    """
    Diffusion Loss (DDPM-style)
    
    Computes the denoising objective:
    L = E[||ε - ε_θ(x_t, t)||²]
    """
    
    def __init__(
        self, 
        num_steps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02
    ):
        super().__init__()
        
        self.num_steps = num_steps
        
        # Register noise schedule
        betas = self._make_beta_schedule(num_steps, beta_start, beta_end)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', 1 - betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(1 - betas, dim=0))
    
    def _make_beta_schedule(
        self, 
        n_timestep: int, 
        beta_start: float, 
        beta_end: float
    ) -> torch.Tensor:
        """Create linear beta schedule"""
        return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, n_timestep) ** 2
    
    def forward(
        self, 
        predicted_noise: torch.Tensor, 
        true_noise: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute diffusion loss
        
        Args:
            predicted_noise: Predicted noise
            true_noise: True noise added
            
        Returns:
            loss: Scalar loss value
        """
        loss = F.mse_loss(predicted_noise, true_noise)
        return loss
    
    def compute(
        self, 
        model: nn.Module, 
        x: torch.Tensor, 
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss with model
        
        Args:
            model: Noise prediction model
            x: Condition features
            action: Clean actions
            
        Returns:
            loss: Scalar loss value
        """
        B = x.shape[0]
        device = x.device
        
        # Sample random timestep
        t = torch.randint(0, self.num_steps, (B,), device=device).long()
        
        # Sample noise
        noise = torch.randn_like(action)
        
        # Add noise
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1)
        noisy_action = torch.sqrt(alpha_cumprod_t) * action + \
                       torch.sqrt(1 - alpha_cumprod_t) * noise
        
        # Predict noise
        predicted_noise = model(x, t.float() / self.num_steps, noisy_action)
        
        # Compute loss
        loss = self.forward(predicted_noise, noise)
        
        return loss


class ActionLoss(nn.Module):
    """
    Direct Action Loss (MSE/Smooth L1)
    
    For simple regression-based action prediction.
    """
    
    def __init__(self, loss_type: str = 'mse'):
        super().__init__()
        self.loss_type = loss_type
        
        if loss_type == 'mse':
            self.loss_fn = F.mse_loss
        elif loss_type == 'l1':
            self.loss_fn = F.l1_loss
        elif loss_type == 'smooth_l1':
            self.loss_fn = F.smooth_l1_loss
        elif loss_type == 'huber':
            self.loss_fn = F.huber_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(
        self, 
        predicted: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute action loss
        
        Args:
            predicted: Predicted actions
            target: Target actions
            
        Returns:
            loss: Scalar loss value
        """
        loss = self.loss_fn(predicted, target)
        return loss


class WeightedActionLoss(nn.Module):
    """
    Weighted Action Loss
    
    Applies different weights to different action dimensions.
    Useful when some dimensions are more important than others.
    """
    
    def __init__(
        self, 
        weights: Optional[torch.Tensor] = None,
        loss_type: str = 'mse'
    ):
        super().__init__()
        
        self.base_loss = ActionLoss(loss_type=loss_type)
        self.register_buffer('weights', weights)
    
    def forward(
        self, 
        predicted: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted action loss
        
        Args:
            predicted: Predicted actions (B, T, D)
            target: Target actions (B, T, D)
            
        Returns:
            loss: Scalar loss value
        """
        if self.weights is None:
            return self.base_loss(predicted, target)
        
        # Apply weights
        diff = predicted - target
        weighted_diff = diff * self.weights.view(1, 1, -1)
        
        if self.base_loss.loss_type == 'mse':
            loss = (weighted_diff ** 2).mean()
        else:
            loss = weighted_diff.abs().mean()
        
        return loss


def build_loss(config: Dict[str, Any]) -> nn.Module:
    """
    Build loss function from config
    
    Args:
        config: Loss configuration
        
    Returns:
        Loss module
    """
    loss_type = config.get('type', 'mse').lower()
    
    if loss_type == 'flow_matching':
        return FlowMatchingLoss(config)
    elif loss_type == 'diffusion':
        return DiffusionLoss(
            num_steps=config.get('num_steps', 100),
            beta_start=config.get('beta_start', 1e-4),
            beta_end=config.get('beta_end', 0.02)
        )
    elif loss_type == 'weighted_mse':
        weights = config.get('weights', None)
        if weights is not None:
            weights = torch.tensor(weights)
        return WeightedActionLoss(weights=weights, loss_type='mse')
    else:
        return ActionLoss(loss_type=loss_type)
