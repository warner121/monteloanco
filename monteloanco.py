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


class Template():
    
    # define a mask for forbidden transitions
    MASK = torch.tensor([
        [ False,  True,  True,  True,  True,  True,  True,  True, ], # [full-paid, current, 30 days late, 60 days late, ..., charged-off]
        [ False, False, False,  True,  True,  True,  True,  True, ],
        [ False, False, False, False,  True,  True,  True,  True, ],
        [ False, False, False, False, False,  True,  True,  True, ],
        [ False, False, False, False, False, False,  True,  True, ],
        [ False, False, False, False, False, False, False,  True, ],
        [ False, False, False, False, False, False, False, False, ],
        [  True,  True,  True,  True,  True,  True,  True, False, ],])

    # define a hand-crafted matrix for demonstration purposes only
    DEMO = torch.tensor([
            [1.,    0.,   0.,    0.,  0.,  0.,  0.,  0., ], # [full-paid, current, 30 days late, 60 days late, ..., charged-off]
            [0.006, 0.96, 0.034, 0.,  0.,  0.,  0.,  0., ],
            [0.,    0.2,  0.2,   0.6, 0.,  0.,  0.,  0., ],
            [0.,    0.2,  0.,    0.2, 0.6, 0.,  0.,  0., ],
            [0.,    0.2,  0.,    0.,  0.2, 0.6, 0.,  0., ],
            [0.,    0.2,  0.,    0.,  0.,  0.2, 0.6, 0., ],
            [0.,    0.2,  0.,    0.,  0.,  0.,  0.2, 0.6,],
            [0.,    0.,   0.,    0.,  0.,  0.,  0.,  1., ],])


def tmat_reshape(tmat, weight1, bias1, device):
    
    #tmat = model.linear1(tmat)
    #tmat = model.linear2(F.relu(tmat))
    tmat = torch.matmul(tmat, weight1.T) + bias1
    tmat = tmat.reshape(-1, 8, 8)
    tmat = tmat.masked_fill(Template.MASK.to(device), float('-inf'))
    tmat = F.softmax(tmat, dim=-1)
    tmat = torch.nan_to_num(tmat, nan=0.0)
    return tmat

    
class Model(PyroModule):
    
    def __init__(self, embedding_size, device='cuda:0', scaling_factor=1_000_000):
        super().__init__()
        self.embedding_size = embedding_size
        self.device = device
        self.scaling_factor = scaling_factor # scale to make the gradients more manageable ($500 becomes 0.5 etc.)
        
        #self.linear1 = torch.nn.Linear(self.embedding_size, 64)
        #self.linear2 = torch.nn.Linear(64, 64)

    def forward(self, batch_id, batch_idx, installments, loan_amnt, int_rate, pymnts=None, num_timesteps=None, demo=False):

        # transpose the input tensors to make stacking/indexing slighly easier
        installments = installments / self.scaling_factor
        loan_amnt = loan_amnt / self.scaling_factor

        batch_size=len(batch_idx)
        if torch.is_tensor(pymnts): 
            pymnts = pymnts.T / self.scaling_factor
            num_timesteps = pymnts.shape[0]
        elif not num_timesteps:
            num_timesteps = 36

        # initalise amortisation
        interest_paid = torch.zeros((1, batch_size)).to(self.device)
        principal_paid = torch.zeros((1, batch_size)).to(self.device)
        
        # initalise other variables
        balances = loan_amnt.clone().unsqueeze(0)
        interest_owed = torch.zeros((1, batch_size)).to(self.device)
        sim_pymnts = torch.zeros((1, batch_size)).to(self.device)
        hidden_states = torch.ones((1, batch_size), dtype=torch.int32).to(self.device)

        # define the embedding and linear terms to translate embedding to transition matrix
        embeddings = pyro.param(f"embeddings_{batch_id}", torch.randn(batch_size, self.embedding_size).to(self.device))
        weight1 = pyro.param(f"model.weight1_{batch_id}", torch.randn(64, self.embedding_size).to(self.device) * 0.1)
        bias1 = pyro.param(f"model.bias1_{batch_id}", torch.randn(64).to(self.device) * 0.1)
        tmat = tmat_reshape(embeddings, weight1, bias1, self.device)

        # overwrite with the demo matrix if demo
        if demo: tmat = Template.DEMO.unsqueeze(0).repeat(batch_size, 1, 1).to(self.device)
        
        with pyro.plate(f"batch_{batch_id}", batch_size, dim=-1):
            for t in range(1, num_timesteps + 1):

                # add interest to the balance
                prev_balances = balances[t - 1]
                interest_owed += prev_balances * int_rate / 1200
                interest_owed = interest_owed.squeeze(0)
        
                # perform the monte-carlo step
                new_hidden_states = pyro.sample(f"hidden_state_{batch_id}_{t}", dist.Categorical(tmat[batch_idx, hidden_states[t - 1]]))
                   
                # calculate the amount that must have been paid to prompt the status update, where the loan has not been charged off, else 0
                # e.g. a change from 3 month's delinquent up to date implies (3 - 0 + 1)
                new_sim_pymnts = torch.where(
                    new_hidden_states < 7,
                    (hidden_states[t - 1] - new_hidden_states + 1) * installments,
                    torch.zeros(batch_size).to(self.device)
                )
                
                # overwrite implied payment with the balance where loan has been fully paid
                new_sim_pymnts = torch.where(
                    new_hidden_states == 0,
                    prev_balances + interest_owed,
                    new_sim_pymnts
                )

                # ensure interest is paid first
                interest_payment = torch.minimum(new_sim_pymnts, interest_owed)
                principal_payment = torch.clamp(new_sim_pymnts - interest_payment, min=0)
                new_balances = torch.clamp(prev_balances - principal_payment, min=0)
                interest_owed = interest_owed - interest_payment
                
                # append new timestep to histories
                balances = torch.cat((balances, new_balances.unsqueeze(0)), dim=0)
                hidden_states = torch.cat((hidden_states, new_hidden_states.unsqueeze(0)), dim=0)
                sim_pymnts = torch.cat((sim_pymnts, new_sim_pymnts.unsqueeze(0)), dim=0)
                interest_paid = torch.cat((interest_paid, interest_payment.unsqueeze(0)), dim=0)
                principal_paid = torch.cat((principal_paid, principal_payment.unsqueeze(0)), dim=0)
                
                # Observation model (noisy measurement of hidden state)
                if torch.is_tensor(pymnts): pyro.sample(f"obs_{batch_id}_{t}", dist.Normal(sim_pymnts[1:t].sum(0), 100. / self.scaling_factor),
                    obs=pymnts[0:t - 1].sum(0)) # pymnts is 1 shorter than the simulated vectors as the origin is omitted

        return hidden_states[1:], sim_pymnts[1:] * self.scaling_factor, interest_paid[1:] * self.scaling_factor, principal_paid[1:] * self.scaling_factor


class Guide(PyroModule):
    
    def __init__(self, embedding_size, device='cuda:0'):
        super().__init__()
        self.embedding_size = embedding_size
        self.device = device

        #self.linear1 = torch.nn.Linear(self.embedding_size, 64)
        #self.linear2 = torch.nn.Linear(64, 64)
        
    def forward(self, batch_id, batch_idx, installments, loan_amnt, int_rate, pymnts):
        
        # transpose the input tensors to make stacking/indexing slighly easier
        pymnts = pymnts.T
    
        # determine the shape of the inputs
        num_timesteps = pymnts.shape[0]
        batch_size = pymnts.shape[1]

        # define the embedding and linear terms to translate embedding to transition matrix
        tmat_prior = pyro.param(
            f"tmat_prior_{batch_id}", 
            Template.DEMO.unsqueeze(0).repeat(batch_size, 1, 1).to(self.device), 
            constraint=torch.distributions.constraints.positive)

        with pyro.plate(f"batch_{batch_id}", batch_size, dim=-1):

            # Variational posterior for the initial hidden state
            hidden_states = torch.ones(batch_size, dtype=torch.int32).to(self.device)
        
            for t in range(1, num_timesteps + 1):
                
                # Variational posterior for each hidden state
                hidden_states = pyro.sample(f"hidden_state_{batch_id}_{t}", dist.Categorical(tmat_prior[batch_idx, hidden_states]))
