import torch
import torch.nn.functional as F
import pyro
import pyro.distributions as dist

from pyro.nn import PyroModule
from torch.utils.data import BatchSampler
from collections import defaultdict


class GroupedBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, grouper='pymnt'):

        # Group indices by tensor length
        self.length_to_indices = defaultdict(list)
        for idx in range(len(dataset)):
            tensor_length = len(dataset[idx][grouper])
            self.length_to_indices[tensor_length].append(idx)

        # Create batches within each group
        self.batches = []
        for length, indices in self.length_to_indices.items():
            for i in range(0, len(indices), batch_size):
                self.batches.append(indices[i:i + batch_size])

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)
        

class Model(PyroModule):
    
    def __init__(self, input_size, embedding_size, device='cuda:0', scaling_factor=1_000_000):
        super().__init__()
        self.device = device
        self.scaling_factor = scaling_factor # scale to make the gradients more manageable ($500 becomes 0.5 etc.)

        # define the embedding and linear terms to translate embedding to transition matrix
        self.embeddings = pyro.param("embeddings", torch.randn(input_size, 64).to(self.device))
        self.linear1 = torch.nn.Linear(embedding_size, 64)
        self.linear2 = torch.nn.Linear(64, 64)

        # define a hand-crafted matrix for demonstration purposes only
        self.tmat_mask = torch.tensor([
            [ False,  True,  True,  True,  True,  True,  True,  True, ], # [full-paid, current, 30 days late, 60 days late, ..., charged-off]
            [ False, False, False,  True,  True,  True,  True,  True, ],
            [ False, False, False, False,  True,  True,  True,  True, ],
            [ False, False, False, False, False,  True,  True,  True, ],
            [ False, False, False, False, False, False,  True,  True, ],
            [ False, False, False, False, False, False, False,  True, ],
            [ False, False, False, False, False, False, False, False, ],
            [  True,  True,  True,  True,  True,  True,  True, False, ],]).to(self.device)
        
        # define a hand-crafted matrix for demonstration purposes only
        self.tmat_demo = torch.tensor([
            [1.,    0.,   0.,    0.,  0.,  0.,  0.,  0., ], # [full-paid, current, 30 days late, 60 days late, ..., charged-off]
            [0.006, 0.96, 0.034, 0.,  0.,  0.,  0.,  0., ],
            [0.,    0.2,  0.2,   0.6, 0.,  0.,  0.,  0., ],
            [0.,    0.2,  0.,    0.2, 0.6, 0.,  0.,  0., ],
            [0.,    0.2,  0.,    0.,  0.2, 0.6, 0.,  0., ],
            [0.,    0.2,  0.,    0.,  0.,  0.2, 0.6, 0., ],
            [0.,    0.2,  0.,    0.,  0.,  0.,  0.2, 0.6,],
            [0.,    0.,   0.,    0.,  0.,  0.,  0.,  1., ],]).to(self.device)

    def _idx_to_tmat(self, idx, batch_size):
        
        tmat = self.embeddings[idx]
        tmat = tmat.reshape(batch_size, 8, 8)
        tmat = tmat.masked_fill(self.tmat_mask, float('-inf'))
        tmat = F.softmax(tmat, dim=-1)
        tmat = torch.nan_to_num(tmat, nan=0.0)
        
        return tmat
    
    def forward(self, batchidx, idx, installments, loan_amnt, int_rate, pymnts=None, num_timesteps=None, demo=False):

        # transpose the input tensors to make stacking/indexing slighly easier
        installments = installments / self.scaling_factor
        loan_amnt = loan_amnt / self.scaling_factor

        batch_size=len(idx)
        if torch.is_tensor(pymnts): 
            pymnts = pymnts.T / self.scaling_factor
            num_timesteps = pymnts.shape[0]
        elif not num_timesteps:
            num_timesteps = 36

        # initalise amortisation
        interest_paid = torch.zeros((1, batch_size)).to(self.device)
        principal_paid = torch.zeros((1, batch_size)).to(self.device)
        remaining_balance = loan_amnt.clone()
        
        # initalise other variables
        sim_pymnts = torch.zeros((1, batch_size)).to(self.device)
        hidden_states = torch.ones((1, batch_size), dtype=torch.int32).to(self.device)

        # fetch embeddings and convert into transition matrices
        tmat = self._idx_to_tmat(idx, batch_size)

        # overwrite with the demo matrix if demo
        if demo: tmat = self.tmat_demo.unsqueeze(0).repeat(batch_size, 1, 1)
        
        with pyro.plate("batch", batch_size, dim=-1):
            for t in range(1, num_timesteps + 1):

                # add interest to the balance
                interest_payment = remaining_balance * int_rate / 1200
                remaining_balance += interest_payment

                # perform the monte-carlo step
                new_hidden_states = pyro.sample(f"hidden_state_{batchidx}_{t}", dist.Categorical(tmat[torch.arange(batch_size), hidden_states[t - 1]]))
                   
                # calculate the amount that must have been paid to prompt the status update, where the loan has not been charged off, else 0
                # e.g. a change from 3 month's delinquent up to date implies (3 - 0 + 1)
                new_sim_pymnts = torch.where(
                    new_hidden_states < 7,
                    (hidden_states[t - 1] - new_hidden_states + 1) * installments, # installments is 1 shorter than the simulated vectors as the origin is omitted
                    torch.zeros(batch_size).to(self.device)
                )
                
                # overwrite implied payment with the balance where loan has been fully paid
                new_sim_pymnts = torch.where(
                    new_hidden_states == 0,
                    remaining_balance,
                    new_sim_pymnts
                )

                # calculate interest and principal
                interest_payment = torch.where(
                    new_sim_pymnts == 0, 
                    0, 
                    interest_payment)
                principal_payment = new_sim_pymnts - interest_payment

                # ensure we don't overpay on the last payment
                remaining_balance -= new_sim_pymnts
                
                # append new timestep to histories
                hidden_states = torch.cat((hidden_states, new_hidden_states.unsqueeze(0)), dim=0)
                sim_pymnts = torch.cat((sim_pymnts, new_sim_pymnts.unsqueeze(0)), dim=0)
                interest_paid = torch.cat((interest_paid, interest_payment.unsqueeze(0)), dim=0)
                principal_paid = torch.cat((principal_paid, principal_payment.unsqueeze(0)), dim=0)
                
                # Observation model (noisy measurement of hidden state)
                if torch.is_tensor(pymnts): pyro.sample(f"obs_{batchidx}_{t}", dist.Normal(sim_pymnts[1:t].sum(0), 1_000. / self.scaling_factor),
                    obs=pymnts[0:t - 1].sum(0)) # pymnts is 1 shorter than the simulated vectors as the origin is omitted

        return hidden_states[1:], sim_pymnts[1:] * self.scaling_factor, interest_paid[1:] * self.scaling_factor, principal_paid[1:] * self.scaling_factor


class Guide(PyroModule):
    
    def __init__(self, device='cuda:0'):
        super().__init__()
        self.device = device
        
        # define a hand-crafted matrix for demonstration purposes only
        self.tmat_mask = torch.tensor([
            [ False,  True,  True,  True,  True,  True,  True,  True, ], # [full-paid, current, 30 days late, 60 days late, ..., charged-off]
            [ False, False, False,  True,  True,  True,  True,  True, ],
            [ False, False, False, False,  True,  True,  True,  True, ],
            [ False, False, False, False, False,  True,  True,  True, ],
            [ False, False, False, False, False, False,  True,  True, ],
            [ False, False, False, False, False, False, False,  True, ],
            [ False, False, False, False, False, False, False, False, ],
            [  True,  True,  True,  True,  True,  True,  True, False, ],]).to(self.device)
        
    def forward(self, batchidx, idx, installments, loan_amnt, int_rate, pymnts):
        
        # transpose the input tensors to make stacking/indexing slighly easier
        pymnts = pymnts.T
    
        # determine the shape of the inputs
        num_timesteps = pymnts.shape[0]
        batch_size = pymnts.shape[1]
    
        with pyro.plate("batch", batch_size, dim=-1):
    
            # Variational parameters for the hidden states
            tmat_prior = pyro.param(f'tmat_prior_{batchidx}',
                dist.Normal(
                    torch.zeros(batch_size, 64).to(self.device),
                    torch.ones(batch_size, 64).to(self.device)))

            tmat_prior = tmat_prior.reshape(batch_size, 8, 8)
            tmat_prior = tmat_prior.masked_fill(self.tmat_mask, float('-inf'))
            tmat_prior = F.softmax(tmat_prior, dim=-1)
            tmat_prior = torch.nan_to_num(tmat_prior, nan=0.0)

            # Variational posterior for the initial hidden state
            hidden_states = torch.ones(batch_size, dtype=torch.int32).to(self.device)
        
            for t in range(1, num_timesteps + 1):
                # Variational posterior for each hidden state
                hidden_states = pyro.sample(f"hidden_state_{batchidx}_{t}", dist.Categorical(tmat_prior[torch.arange(batch_size), hidden_states]))
