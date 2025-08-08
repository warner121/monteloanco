import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist

from pyro.nn import PyroModule
from torch.utils.data import BatchSampler
from collections import defaultdict


class GroupedBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, grouper='n_report_d'):

        # Group indices by tensor length
        self.length_to_indices = defaultdict(list)
        for idx in range(len(dataset)):
            n_report_d = dataset[idx][grouper]
            self.length_to_indices[n_report_d].append(idx)

        # Create batches within each group
        self.batches = []
        for length, indices in self.length_to_indices.items():
            for i in range(0, len(indices), batch_size):
                self.batches.append(indices[i:i + batch_size])

    def __iter__(self):
        return iter(self.batches)


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
        [1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000], # [full-paid, current, 30 days late, 60 days late, ..., charged-off]
        [0.0087, 0.9829, 0.0084, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.1541, 0.4723, 0.1541, 0.2195, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.1439, 0.3408, 0.1439, 0.1439, 0.2274, 0.0000, 0.0000, 0.0000],
        [0.1310, 0.2481, 0.1310, 0.1310, 0.1310, 0.2281, 0.0000, 0.0000],
        [0.1189, 0.1820, 0.1189, 0.1189, 0.1189, 0.1189, 0.2236, 0.0000],
        [0.1089, 0.1352, 0.1089, 0.1089, 0.1089, 0.1089, 0.1089, 0.2111],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000]])


class TransitionMatrixNet(nn.Module):
    """Neural network to generate transition matrices from embeddings."""
    
    def __init__(self, embedding_size, hidden_size=64, device='cuda:0'):
        super().__init__()
        self.device = device
        self.linear1 = nn.Linear(embedding_size, hidden_size)
        
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


def model(batch_id, batch_idx, installments, loan_amnt, int_rate, 
          total_pre_chargeoff=None, num_timesteps=None, demo=False,
          embedding_size=3, device='cuda:0', scaling_factor=1_000_000,
          transition_net=None):
    """
    Pyro model for loan state transitions and payments.
    """
    
    # Scale inputs
    installments = installments / scaling_factor
    loan_amnt = loan_amnt / scaling_factor
    
    # Determine shape of batch
    batch_size = len(batch_idx)
    if not num_timesteps: 
        num_timesteps = 60

    # Initialize amortization
    interest_paid = torch.zeros((1, batch_size)).to(device)
    principal_paid = torch.zeros((1, batch_size)).to(device)
    
    # Initialize other variables
    balances = loan_amnt.clone().unsqueeze(0)
    interest_owed = torch.zeros((1, batch_size)).to(device)
    sim_pymnts = torch.zeros((1, batch_size)).to(device)
    hidden_states = torch.ones((1, batch_size), dtype=torch.int32).to(device)

    # Define embeddings as Pyro parameters
    embeddings = pyro.param(
        f"embeddings_{batch_id}", 
        torch.randn(batch_size, embedding_size).to(device)
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
        for t in range(1, num_timesteps + 1):

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
                torch.zeros(batch_size).to(device)
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
            
        # Observation model (noisy measurement of hidden state)
        if torch.is_tensor(total_pre_chargeoff):                 
            pyro.sample(
                f"obs_{batch_id}_{t}", 
                dist.Normal(sim_pymnts[1:t].sum(0), 100. / scaling_factor),
                obs=total_pre_chargeoff / scaling_factor
            )

    return (
        hidden_states[1:], 
        sim_pymnts[1:] * scaling_factor, 
        interest_paid[1:] * scaling_factor, 
        principal_paid[1:] * scaling_factor
    )


def guide(batch_id, batch_idx, installments, loan_amnt, int_rate, 
          total_pre_chargeoff, num_timesteps, device='cuda:0'):
    """
    Pyro guide (variational posterior) for loan state transitions.
    """
    
    # Determine the shape of the inputs
    batch_size = len(batch_idx)

    # Define the transition matrix prior as a parameter
    tmat_prior = pyro.param(
        f"tmat_prior_{batch_id}", 
        Template.DEMO.unsqueeze(0).repeat(batch_size, 1, 1).to(device), 
        constraint=torch.distributions.constraints.positive
    )

    with pyro.plate(f"batch_{batch_id}", batch_size, dim=-1):

        # Variational posterior for the initial hidden state
        hidden_states = torch.ones(batch_size, dtype=torch.int32).to(device)
    
        for t in range(1, num_timesteps + 1):
            
            # Variational posterior for each hidden state
            hidden_states = pyro.sample(
                f"hidden_state_{batch_id}_{t}", 
                dist.Categorical(tmat_prior[batch_idx, hidden_states])
            )
