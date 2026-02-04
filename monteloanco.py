import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist

from pyro.nn import PyroModule
from torch.utils.data import BatchSampler
from collections import defaultdict


class Template():
    
    # Define a mask for forbidden transitions
    MASK = torch.tensor([
        [ False,  True,  True,  True,  True,  True,  True,  True, ], # [full-paid, current, 30 days late, 60 days late, ..., charged-off]
        [ False, False, False,  True,  True,  True,  True,  True, ],
        [ False, False, False, False,  True,  True,  True,  True, ],
        [ False, False, False, False, False,  True,  True,  True, ],
        [ False, False, False, False, False, False,  True,  True, ],
        [ False, False, False, False, False, False, False,  True, ],
        [ False, False, False, False, False, False, False, False, ],
        [  True,  True,  True,  True,  True,  True,  True, False, ]])

    # Define a hand-crafted matrix for demonstration purposes only
    DEMO = torch.tensor([
        [1.000, 0.000, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0,],
        [0.030, 0.960, 0.01, 0.00, 0.00, 0.00, 0.00, 0.0,],
        [0.025, 0.325, 0.05, 0.60, 0.00, 0.00, 0.00, 0.0,],
        [0.020, 0.340, 0.00, 0.04, 0.60, 0.00, 0.00, 0.0,],
        [0.015, 0.355, 0.00, 0.00, 0.03, 0.60, 0.00, 0.0,],
        [0.010, 0.370, 0.00, 0.00, 0.00, 0.02, 0.60, 0.0,],
        [0.005, 0.385, 0.00, 0.00, 0.00, 0.00, 0.01, 0.6,],
        [0.000, 0.000, 0.00, 0.00, 0.00, 0.00, 0.00, 1.0,]])


class TransitionMatrixNet(nn.Module):
    """Neural network to generate transition matrices from embeddings."""
    
    def __init__(self, embedding_size, device='cuda:0'):
        super().__init__()
        self.device = device
        self.linear1 = nn.Linear(embedding_size, 64)
        
        # Register the mask as a buffer so it moves with the module
        self.register_buffer('mask', Template.MASK)
        
    def forward(self, embeddings):
        """
        Transform embeddings into transition matrices.
        
        Args:
            embeddings: Tensor of shape (batch_size, embedding_size)
            
        Returns:
            Tensor of shape (batch_size, 8, 8) representing transition matrices
        """
        # Apply linear transformation
        tmat = self.linear1(embeddings)
        
        # Reshape to transition matrix format
        tmat = tmat.reshape(-1, 8, 8)
        
        # Apply mask for forbidden transitions
        tmat = tmat.masked_fill(self.mask, float('-inf'))
        
        # Apply softmax to get valid probability distributions
        tmat = F.softmax(tmat, dim=-1)
        
        # Handle any NaN values
        tmat = torch.nan_to_num(tmat, nan=0.0)
        
        return tmat


def _prepare_num_timesteps(num_timesteps, batch_size, device):
    """
    Helper to normalize num_timesteps to a tensor.
    
    Args:
        num_timesteps: None or tensor of shape (batch_size,)
        batch_size: Size of current batch
        device: Device to place tensor on
        
    Returns:
        Tensor of shape (batch_size,) with dtype torch.long
    """
    if torch.is_tensor(num_timesteps):
        num_timesteps = num_timesteps.to(device)
    else:
        num_timesteps = torch.full((batch_size,), num_timesteps, dtype=torch.long, device=device)

    return num_timesteps


def _apply_timestep_mask(num_timesteps, max_timesteps, target, device):
    """
    Helper to mask tensor elements that exeed the bounds of num_timesteps.
    
    Args:
        num_timesteps: Tensor of shape (batch_size,)
        max_timesteps: Maximum number of timesteps simulated
        target: Full tensor to which mask is applied
        device: Device to place tensor on
    """
    timestep_range = torch.arange(1, max_timesteps + 1, device=device).unsqueeze(1)
    mask = timestep_range <= num_timesteps.unsqueeze(0)
    
    return target * mask


def _extract_last_nonzero_payment(sim_pymnts, num_timesteps, device):
    """
    Extract the last non-zero payment for each loan in the batch.
    
    Args:
        sim_pymnts: Tensor of shape (max_timesteps, batch_size) containing payment history
        num_timesteps: Tensor of shape (batch_size,) indicating number of observed timesteps
        device: Device to place tensor on
        
    Returns:
        Tensor of shape (batch_size,) containing the last non-zero payment for each loan
    """
    batch_size = sim_pymnts.shape[1]
    max_timesteps = sim_pymnts.shape[0]
    
    # Create indices for the last observed timestep for each loan
    # num_timesteps is 1-indexed, so these are direct indices into sim_pymnts
    last_indices = num_timesteps.clamp(max=max_timesteps) - 1
    
    # Gather the last payment for each loan
    # sim_pymnts is (max_timesteps, batch_size), we want sim_pymnts[last_indices[i], i]
    batch_indices = torch.arange(batch_size, device=device)
    last_payments = sim_pymnts[last_indices, batch_indices]
    
    return last_payments
            

def model(batch_id, batch_idx, installments, loan_amnt, int_rate, 
          total_pre_chargeoff=None, last_pymnt_amnt=None, num_timesteps=60, 
          demo=False, embedding_size=3, device='cuda:0', scaling_factor=1_000_000,
          transition_net=None):
    """
    Pyro model for loan state transitions and payments.
    
    Args:
        batch_id: Identifier for the batch
        batch_idx: Indices of loans in the batch
        installments: Monthly installment amounts (batch_size,)
        loan_amnt: Initial loan amounts (batch_size,)
        int_rate: Annual interest rates (batch_size,)
        total_pre_chargeoff: Observed total payments before chargeoff (batch_size,), optional
        last_pymnt_amnt: Observed last payment amount (batch_size,), optional
        num_timesteps: Number of observed timesteps per loan (batch_size,), optional
        demo: Whether to use demo transition matrix
        embedding_size: Size of loan embeddings
        device: Device to run on
        scaling_factor: Factor to scale monetary amounts
        transition_net: Neural network for generating transition matrices
    """
    
    # Scale inputs
    installments = installments / scaling_factor
    loan_amnt = loan_amnt / scaling_factor
    
    # Determine shape of batch
    batch_size = len(batch_idx)
    
    # Normalize num_timesteps
    num_timesteps = _prepare_num_timesteps(num_timesteps, batch_size, device)
    max_timesteps = num_timesteps.max().item()

    # Initialize time series accumulators with shape (1, batch_size)
    # These will grow to (max_timesteps+1, batch_size) via concatenation
    interest_paid = torch.zeros((1, batch_size), device=device)
    principal_paid = torch.zeros((1, batch_size), device=device)
    balances = loan_amnt.clone().unsqueeze(0)  # (batch_size,) -> (1, batch_size)
    interest_owed = torch.zeros((1, batch_size), device=device)
    sim_pymnts = torch.zeros((1, batch_size), device=device)
    hidden_states = torch.ones((1, batch_size), dtype=torch.int32, device=device)

    # Define embeddings as Pyro parameters
    embeddings = pyro.param(
        f"embeddings_{batch_id}", 
        torch.randn(batch_size, embedding_size, device=device)
    )
    
    # Create and register the neural network module
    transition_net = pyro.module(
        f"transition_net",
        transition_net
    )
    
    # Generate transition matrices
    if demo:
        tmat = Template.DEMO.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    else:
        tmat = transition_net(embeddings)
    
    with pyro.plate(f"batch_{batch_id}", batch_size, dim=-1):
        for t in range(1, max_timesteps + 1):

            # Add interest to the balance
            prev_balances = balances[t - 1]
            interest_owed += prev_balances * int_rate / 1200
            interest_owed = interest_owed.squeeze(0)
    
            # Perform the monte-carlo step
            new_hidden_states = pyro.sample(
                f"hidden_state_{batch_id}_{t}", 
                dist.Categorical(tmat[batch_idx, hidden_states[t - 1]])
            )
               
            # Calculate the amount that must have been paid to prompt the status update
            new_sim_pymnts = torch.where(
                new_hidden_states < 7,
                (hidden_states[t - 1] - new_hidden_states + 1) * installments,
                torch.zeros(batch_size, device=device)
            )
            
            # Overwrite implied payment with the balance where loan has been fully paid
            new_sim_pymnts = torch.where(
                new_hidden_states == 0,
                prev_balances + interest_owed,
                new_sim_pymnts
            )

            # Ensure interest is paid first
            interest_payment = torch.minimum(new_sim_pymnts, interest_owed)
            principal_payment = torch.clamp(new_sim_pymnts - interest_payment, min=0)
            new_balances = torch.clamp(prev_balances - principal_payment, min=0)
            interest_owed = interest_owed - interest_payment
            
            # Append new timestep to histories
            balances = torch.cat((balances, new_balances.unsqueeze(0)), dim=0)
            hidden_states = torch.cat((hidden_states, new_hidden_states.unsqueeze(0)), dim=0)
            sim_pymnts = torch.cat((sim_pymnts, new_sim_pymnts.unsqueeze(0)), dim=0)
            interest_paid = torch.cat((interest_paid, interest_payment.unsqueeze(0)), dim=0)
            principal_paid = torch.cat((principal_paid, principal_payment.unsqueeze(0)), dim=0)

        # mask unobserved timesteps
        masked_pymnts = _apply_timestep_mask(num_timesteps, max_timesteps, target=sim_pymnts[1:], device=device)

        # Observation model (noisy measurement of hidden state)
        if torch.is_tensor(total_pre_chargeoff):
            predicted_total = masked_pymnts.sum(0)

            # Scale variance by log of number of timesteps (more aggressive than sqrt)
            base_std = 100. / scaling_factor
            total_std = base_std * torch.log(num_timesteps.float())
            
            pyro.sample(
                f"obs_total_{batch_id}", 
                dist.Normal(predicted_total, total_std),
                obs=total_pre_chargeoff / scaling_factor
            )
        
        # Additional observation for last payment amount
        if torch.is_tensor(last_pymnt_amnt):
            predicted_last = masked_pymnts.max(0)[0]
            
            # Standard deviation for last payment observation
            # Could be constant or scaled based on loan characteristics
            base_std = 50. / scaling_factor
            last_std = base_std * torch.log(num_timesteps.float())
            
            pyro.sample(
                f"obs_last_{batch_id}",
                dist.Normal(predicted_last, last_std),
                obs=last_pymnt_amnt / scaling_factor
            )

    return (
        hidden_states[1:], 
        sim_pymnts[1:] * scaling_factor, 
        interest_paid[1:] * scaling_factor, 
        principal_paid[1:] * scaling_factor
    )


def guide(batch_id, batch_idx, installments, loan_amnt, int_rate, 
          total_pre_chargeoff, last_pymnt_amnt=None, num_timesteps=60, 
          device='cuda:0'):
    """
    Pyro guide (variational posterior) for loan state transitions.
    
    Args:
        batch_id: Identifier for the batch
        batch_idx: Indices of loans in the batch
        installments: Monthly installment amounts (batch_size,)
        loan_amnt: Initial loan amounts (batch_size,)
        int_rate: Annual interest rates (batch_size,)
        total_pre_chargeoff: Observed total payments before chargeoff (batch_size,)
        last_pymnt_amnt: Observed last payment amount (batch_size,), optional
        num_timesteps: Number of observed timesteps per loan (batch_size,)
        device: Device to run on
    """
    
    batch_size = len(batch_idx)
    
    # Normalize num_timesteps (same helper function)
    num_timesteps = _prepare_num_timesteps(num_timesteps, batch_size, device)
    max_timesteps = num_timesteps.max().item()

    # Define the transition matrix prior as a parameter (in logit space)
    # For Categorical distribution with multiple outcomes, we just need unnormalized log probabilities
    # Starting from probabilities, we can simply take the log (softmax will renormalize)
    demo_logits = torch.log(Template.DEMO.clamp(min=1e-8))  # Clamp to avoid log(0)
    tmat_logits = pyro.param(
        f"tmat_prior_{batch_id}", 
        demo_logits.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    )

    with pyro.plate(f"batch_{batch_id}", batch_size, dim=-1):
        hidden_states = torch.ones(batch_size, dtype=torch.int32, device=device)
    
        for t in range(1, max_timesteps + 1):
            hidden_states = pyro.sample(
                f"hidden_state_{batch_id}_{t}", 
                dist.Categorical(logits=tmat_logits[batch_idx, hidden_states])
            )