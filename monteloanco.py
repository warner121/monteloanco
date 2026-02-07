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
    
    @staticmethod
    def get_demo_logits(device='cuda:0'):
        """
        Get initial logits from the demo transition matrix.
        
        Args:
            device: Device to place tensor on
            
        Returns:
            Flattened logits tensor of shape (64,)
        """
        return torch.log(Template.DEMO.clamp(min=1e-8)).flatten().to(device)
    
    @staticmethod
    def initialize_tmat_logits(batch_id, batch_size, device='cuda:0'):
        """
        Initialize transition matrix logits as a Pyro parameter.
        
        Args:
            batch_id: Identifier for the batch
            batch_size: Number of loans in batch
            device: Device to place tensor on
            
        Returns:
            Pyro parameter of shape (batch_size, 64)
        """
        demo_logits = Template.get_demo_logits(device)
        return pyro.param(
            f"tmat_logits_{batch_id}", 
            demo_logits.unsqueeze(0).repeat(batch_size, 1)
        )
    
    @staticmethod
    def logits_to_tmat(logits, device='cuda:0'):
        """
        Transform logits into masked logit matrices for Categorical distribution.
        
        Args:
            logits: Tensor of shape (batch_size, 64) representing flattened 8x8 matrices
            device: Device to place tensor on
            
        Returns:
            Tensor of shape (batch_size, 8, 8) with forbidden transitions set to -inf
        """
        # Reshape to transition matrix format
        tmat_logits = logits.reshape(-1, 8, 8)
        
        # Apply mask for forbidden transitions (set to -inf)
        mask = Template.MASK.to(device)
        tmat_logits = tmat_logits.masked_fill(mask, float('-inf'))
        
        return tmat_logits
    
    @staticmethod
    def get_tmat(batch_id, batch_size, demo=False, device='cuda:0'):
        """
        Get transition matrix logits for the batch.
        
        Args:
            batch_id: Identifier for the batch
            batch_size: Number of loans in batch
            demo: Whether to use demo transition matrix
            device: Device to place tensor on
            
        Returns:
            Tensor of shape (batch_size, 8, 8) representing transition matrix logits
        """
        if demo:
            # Convert demo probabilities to logits
            demo_logits = torch.log(Template.DEMO.clamp(min=1e-8))
            return demo_logits.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        else:
            tmat_logits = Template.initialize_tmat_logits(batch_id, batch_size, device)
            return Template.logits_to_tmat(tmat_logits, device)


class Portfolio:
    """Manages batch of loans with vectorized operations"""
    
    def __init__(self, batch_idx, loan_amnt, installments, int_rate, 
                 num_timesteps=60, total_pre_chargeoff=None, 
                 last_pymnt_amnt=None, device='cuda:0', scaling_factor=1_000_000):
        """
        Initialize a portfolio of loans.
        
        Args:
            batch_idx: Indices of loans in the batch
            loan_amnt: Initial loan amounts (batch_size,)
            installments: Monthly installment amounts (batch_size,)
            int_rate: Annual interest rates (batch_size,)
            num_timesteps: Number of observed timesteps per loan (batch_size,) or int, optional
            total_pre_chargeoff: Observed total payments before chargeoff (batch_size,), optional
            last_pymnt_amnt: Observed last payment amount (batch_size,), optional
            device: Device to run on
            scaling_factor: Factor to scale monetary amounts
        """
        # Loan characteristics (scaled)
        self.batch_idx = batch_idx
        self.batch_size = len(batch_idx)
        self.loan_amnt = loan_amnt / scaling_factor
        self.installments = installments / scaling_factor
        self.int_rate = int_rate
        self.device = device
        self.scaling_factor = scaling_factor
        
        # Normalize num_timesteps to tensor
        if torch.is_tensor(num_timesteps):
            self.num_timesteps = num_timesteps.to(device)
        else:
            self.num_timesteps = torch.full((self.batch_size,), num_timesteps, dtype=torch.long, device=device)
        
        self.max_timesteps = self.num_timesteps.max().item()
        
        # Observation data (scaled if provided)
        self.total_pre_chargeoff = total_pre_chargeoff / scaling_factor if torch.is_tensor(total_pre_chargeoff) else None
        self.last_pymnt_amnt = last_pymnt_amnt / scaling_factor if torch.is_tensor(last_pymnt_amnt) else None
        
        # Current state
        self.current_balances = self.loan_amnt.clone()
        self.current_interest_owed = torch.zeros(self.batch_size, device=device)
        self.current_hidden_states = torch.ones(self.batch_size, dtype=torch.int32, device=device)
        
        # Histories (start with t=0)
        self.balances_history = [self.current_balances.clone()]
        self.interest_paid_history = [torch.zeros(self.batch_size, device=device)]
        self.principal_paid_history = [torch.zeros(self.batch_size, device=device)]
        self.payments_history = [torch.zeros(self.batch_size, device=device)]
        self.hidden_states_history = [self.current_hidden_states.clone()]
    
    def _apply_timestep_mask(self, target):
        """
        Apply mask to tensor elements that exceed the bounds of num_timesteps.
        
        Args:
            target: Tensor of shape (max_timesteps, batch_size) to mask
            
        Returns:
            Masked tensor with same shape
        """
        timestep_range = torch.arange(1, self.max_timesteps + 1, device=self.device).unsqueeze(1)
        mask = timestep_range <= self.num_timesteps.unsqueeze(0)
        return target * mask
        
    def calculate_payment(self, new_hidden_states, old_hidden_states):
        """
        Calculate implied payment amount from state transition.
        
        Args:
            new_hidden_states: New hidden states after transition
            old_hidden_states: Previous hidden states
            
        Returns:
            Tensor of payment amounts (batch_size,)
        """
        payment = torch.where(
            new_hidden_states < 7,
            (old_hidden_states - new_hidden_states + 1) * self.installments,
            torch.zeros(self.batch_size, device=self.device)
        )
        # Full payoff case
        payment = torch.where(
            new_hidden_states == 0,
            self.current_balances + self.current_interest_owed,
            payment
        )
        return payment
        
    def apply_payment(self, payment):
        """
        Apply payment to interest first, then principal.
        
        Args:
            payment: Payment amounts (batch_size,)
            
        Returns:
            Tuple of (interest_paid, principal_paid, new_balance)
        """
        interest_payment = torch.minimum(payment, self.current_interest_owed)
        principal_payment = torch.clamp(payment - interest_payment, min=0)
        new_balance = torch.clamp(self.current_balances - principal_payment, min=0)
        return interest_payment, principal_payment, new_balance
        
    def accrue_interest(self):
        """Add monthly interest to interest_owed"""
        self.current_interest_owed += self.current_balances * self.int_rate / 1200
        
    def step(self, new_hidden_states):
        """
        Execute one timestep given new hidden states (sampled externally by Pyro).
        Updates internal state and appends to histories.
        
        Args:
            new_hidden_states: New hidden states for this timestep (batch_size,)
        """
        # Accrue interest
        self.accrue_interest()
        
        # Calculate payment implied by state transition
        payment = self.calculate_payment(new_hidden_states, self.current_hidden_states)
        
        # Apply payment
        interest_paid, principal_paid, new_balance = self.apply_payment(payment)
        
        # Update current state
        self.current_balances = new_balance
        self.current_interest_owed = self.current_interest_owed - interest_paid
        self.current_hidden_states = new_hidden_states
        
        # Append to histories
        self.balances_history.append(new_balance.clone())
        self.interest_paid_history.append(interest_paid)
        self.principal_paid_history.append(principal_paid)
        self.payments_history.append(payment)
        self.hidden_states_history.append(new_hidden_states.clone())
        
    def get_histories(self):
        """
        Return stacked histories (excluding t=0).
        
        Returns:
            Dictionary with keys: hidden_states, payments, interest_paid, principal_paid
            Each value is a tensor of shape (max_timesteps, batch_size)
        """
        return {
            'hidden_states': torch.stack(self.hidden_states_history[1:]),
            'payments': torch.stack(self.payments_history[1:]),
            'interest_paid': torch.stack(self.interest_paid_history[1:]),
            'principal_paid': torch.stack(self.principal_paid_history[1:])
        }
        
    def get_total_pre_chargeoff(self):
        """
        Return total payments with proper masking.
        
        Returns:
            Tensor of total payments (batch_size,)
        """
        payments = torch.stack(self.payments_history[1:])
        masked_payments = self._apply_timestep_mask(payments)
        return masked_payments.sum(0)
        
    def get_last_payment(self):
        """
        Return last non-zero payment with proper masking.
        
        Returns:
            Tensor of last payments (batch_size,)
        """
        payments = torch.stack(self.payments_history[1:])
        masked_payments = self._apply_timestep_mask(payments)
        return masked_payments.max(0)[0]


def model(batch_id, batch_idx, installments, loan_amnt, int_rate, 
          total_pre_chargeoff=None, last_pymnt_amnt=None, num_timesteps=60, 
          demo=False, device='cuda:0', scaling_factor=1_000_000):
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
        device: Device to run on
        scaling_factor: Factor to scale monetary amounts
    """
    
    batch_size = len(batch_idx)
    
    # Initialize portfolio
    portfolio = Portfolio(
        batch_idx, loan_amnt, installments, int_rate,
        num_timesteps, total_pre_chargeoff, last_pymnt_amnt,
        device, scaling_factor
    )
    
    # Get transition matrix logits
    tmat_logits = Template.get_tmat(batch_id, batch_size, demo=demo, device=device)
    
    with pyro.plate(f"batch_{batch_id}", batch_size, dim=-1):
        for t in range(1, portfolio.max_timesteps + 1):
            # Sample new hidden states
            new_hidden_states = pyro.sample(
                f"hidden_state_{batch_id}_{t}", 
                dist.Categorical(logits=tmat_logits[batch_idx, portfolio.current_hidden_states])
            )
            
            # Update portfolio
            portfolio.step(new_hidden_states)
        
        # Observations
        if torch.is_tensor(total_pre_chargeoff):
            predicted_total = portfolio.get_total_pre_chargeoff()
            base_std = 50. / scaling_factor
            total_std = base_std * torch.sqrt(portfolio.num_timesteps.float())
            pyro.sample(
                f"obs_total_{batch_id}", 
                dist.Normal(predicted_total, total_std),
                obs=total_pre_chargeoff / scaling_factor
            )
        
        if torch.is_tensor(last_pymnt_amnt):
            predicted_last = portfolio.get_last_payment()
            base_std = 50. / scaling_factor
            last_std = base_std * torch.sqrt(portfolio.num_timesteps.float())
            pyro.sample(
                f"obs_last_{batch_id}",
                dist.Normal(predicted_last, last_std),
                obs=last_pymnt_amnt / scaling_factor
            )
    
    # Return scaled histories
    histories = portfolio.get_histories()
    return (
        histories['hidden_states'],
        histories['payments'] * scaling_factor,
        histories['interest_paid'] * scaling_factor,
        histories['principal_paid'] * scaling_factor
    )


def guide(batch_id, batch_idx, installments, loan_amnt, int_rate, 
          total_pre_chargeoff, last_pymnt_amnt=None, num_timesteps=60, 
          demo=False, device='cuda:0', scaling_factor=1_000_000):
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
        demo: Whether to use demo transition matrix
        device: Device to run on
        scaling_factor: Factor to scale monetary amounts
    """
    
    batch_size = len(batch_idx)
    
    # Initialize portfolio (for consistent state management)
    portfolio = Portfolio(
        batch_idx, loan_amnt, installments, int_rate,
        num_timesteps, total_pre_chargeoff, last_pymnt_amnt,
        device, scaling_factor
    )

    # Get transition matrix logits using Template class (aligned with model)
    tmat_logits = Template.get_tmat(batch_id, batch_size, demo=demo, device=device)

    with pyro.plate(f"batch_{batch_id}", batch_size, dim=-1):
        for t in range(1, portfolio.max_timesteps + 1):
            new_hidden_states = pyro.sample(
                f"hidden_state_{batch_id}_{t}", 
                dist.Categorical(logits=tmat_logits[batch_idx, portfolio.current_hidden_states])
            )
            
            # Update portfolio state
            portfolio.step(new_hidden_states)