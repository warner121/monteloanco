import torch
import pyro
import pyro.distributions as dist


class Template:
    """
    Pyro-agnostic utilities and canonical transition definitions.
    """

    MASK = torch.tensor([
        [ False,  True,  True,  True,  True,  True,  True,  True ],
        [ False, False, False,  True,  True,  True,  True,  True ],
        [ False, False, False, False,  True,  True,  True,  True ],
        [ False, False, False, False, False,  True,  True,  True ],
        [ False, False, False, False, False, False,  True,  True ],
        [ False, False, False, False, False, False, False,  True ],
        [ False, False, False, False, False, False, False, False ],
        [  True,  True,  True,  True,  True,  True,  True, False ],
    ])

    DEMO_PROBS = torch.tensor([
        [1.000, 0.000, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0],
        [0.030, 0.960, 0.01, 0.00, 0.00, 0.00, 0.00, 0.0],
        [0.025, 0.325, 0.05, 0.60, 0.00, 0.00, 0.00, 0.0],
        [0.020, 0.340, 0.00, 0.04, 0.60, 0.00, 0.00, 0.0],
        [0.015, 0.355, 0.00, 0.00, 0.03, 0.60, 0.00, 0.0],
        [0.010, 0.370, 0.00, 0.00, 0.00, 0.02, 0.60, 0.0],
        [0.005, 0.385, 0.00, 0.00, 0.00, 0.00, 0.01, 0.6],
        [0.000, 0.000, 0.00, 0.00, 0.00, 0.00, 0.00, 1.0],
    ])

    @staticmethod
    def probs_to_logits(probs: torch.Tensor) -> torch.Tensor:
        """
        Convert probabilities to logits, applying the mask.
        """
        logits = torch.log(probs.clamp(min=1e-8))
        return logits.masked_fill(Template.MASK.to(logits.device), float("-inf"))

    @staticmethod
    def apply_mask(logits: torch.Tensor) -> torch.Tensor:
        """
        Apply the canonical mask to logits, automatically matching device.
        """
        return logits.masked_fill(Template.MASK.to(logits.device), float("-inf"))

    @staticmethod
    def batch_logits(logits: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Expand logits to a batch dimension.
        """
        return logits.unsqueeze(0).expand(batch_size, -1, -1)

# Compute DEMO_LOGITS after the class is fully defined
Template.DEMO_LOGITS = Template.probs_to_logits(Template.DEMO_PROBS)


class TransitionMatrixProvider:
    """
    Abstract interface for supplying transition matrix logits.
    """
    def get_logits(self, batch_size, device):
        raise NotImplementedError


class DemoTransition(TransitionMatrixProvider):
    def get_logits(self, batch_id, batch_idx, batch_size, device):
        return Template.batch_logits(
            Template.DEMO_LOGITS.to(device),
            batch_size
        )


class ExternalTransition(TransitionMatrixProvider):
    """
    logits may be:
      - (8, 8)
      - (batch, 8, 8)
    """

    def __init__(self, logits):
        self.logits = logits

    def get_logits(self, batch_id, batch_idx, batch_size, device):
        logits = Template.apply_mask(self.logits.to(device))
        if logits.dim() == 2:
            return logits.unsqueeze(0).expand(batch_size, -1, -1)
        return logits


class LearnedTransition(TransitionMatrixProvider):
    """
    Shared or per-batch learned transition matrix via pyro.param.
    """

    def __init__(self, name, init_logits=None, trainable=True):
        self.name = name
        self.init_logits = init_logits
        self.trainable = trainable

    def get_logits(self, batch_id, batch_idx, batch_size, device):
        if self.init_logits is None:
            raise ValueError("init_logits must be provided for learned transitions")

        flat_init = self.init_logits.flatten().to(device)

        logits_flat = pyro.param(
            f"{self.name}_{batch_id}",
            flat_init.unsqueeze(0).expand(batch_size, -1)
        )

        if not self.trainable:
            logits_flat = logits_flat.detach()

        tmat = logits_flat.view(-1, 8, 8)
        tmat = tmat[batch_idx]
        return Template.apply_mask(tmat)


class Portfolio:
    """Manages batch of loans with vectorized operations"""
    
    def __init__(self, loan_amnt, installments, int_rate,
                 num_timesteps=60, total_pre_chargeoff=None, 
                 last_pymnt_amnt=None, device='cuda:0', scaling_factor=1_000_000):
        """
        Initialize a portfolio of loans.
        
        Args:
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
        self.batch_size = len(loan_amnt)
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
            'payments': torch.stack(self.payments_history[1:]) * self.scaling_factor,
            'interest_paid': torch.stack(self.interest_paid_history[1:]) * self.scaling_factor,
            'principal_paid': torch.stack(self.principal_paid_history[1:]) * self.scaling_factor
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


def model(
    batch_id,
    batch_idx,
    installments,
    loan_amnt,
    int_rate,
    tmat_provider: TransitionMatrixProvider,
    total_pre_chargeoff=None,
    last_pymnt_amnt=None,
    num_timesteps=60,
    device="cuda:0",
    scaling_factor=1_000_000,
):

    batch_size = len(batch_idx)

    portfolio = Portfolio(
        loan_amnt, installments, int_rate,
        num_timesteps, total_pre_chargeoff, last_pymnt_amnt,
        device, scaling_factor
    )

    tmat_logits = tmat_provider.get_logits(batch_id, batch_idx, batch_size, device)

    with pyro.plate(f"batch_{batch_id}", batch_size, dim=-1):

        for t in range(1, portfolio.max_timesteps + 1):
            new_hidden_states = pyro.sample(
                f"h_{batch_id}_{t}",
                dist.Categorical(logits=tmat_logits[torch.arange(batch_size, device=device), portfolio.current_hidden_states])
            )
            portfolio.step(new_hidden_states)

        if torch.is_tensor(total_pre_chargeoff):
            pred = portfolio.get_total_pre_chargeoff()
            std = (50. / scaling_factor) * torch.sqrt(portfolio.num_timesteps.float())
            pyro.sample(
                f"obs_total_{batch_id}",
                dist.Normal(pred, std),
                obs=total_pre_chargeoff / scaling_factor
            )

        if torch.is_tensor(last_pymnt_amnt):
            pred = portfolio.get_last_payment()
            std = (50. / scaling_factor) * torch.sqrt(portfolio.num_timesteps.float())
            pyro.sample(
                f"obs_last_{batch_id}",
                dist.Normal(pred, std),
                obs=last_pymnt_amnt / scaling_factor
            )

    return portfolio


def guide(
    batch_id,
    batch_idx,
    installments,
    loan_amnt,
    int_rate,
    tmat_provider: TransitionMatrixProvider,
    total_pre_chargeoff=None,
    last_pymnt_amnt=None,
    num_timesteps=60,
    device="cuda:0",
    scaling_factor=1_000_000,
):
    
    batch_size = len(batch_idx)
    
    # Initialize portfolio (for consistent state management)
    portfolio = Portfolio(
        loan_amnt, installments, int_rate,
        num_timesteps, total_pre_chargeoff, last_pymnt_amnt,
        device, scaling_factor
    )

    # Get transition matrix logits using Template class (aligned with model)
    tmat_logits = tmat_provider.get_logits(batch_id, batch_idx, batch_size, device)

    with pyro.plate(f"batch_{batch_id}", batch_size, dim=-1):
        for t in range(1, portfolio.max_timesteps + 1):
            new_hidden_states = pyro.sample(
                f"h_{batch_id}_{t}",
                dist.Categorical(logits=tmat_logits[torch.arange(batch_size, device=device), portfolio.current_hidden_states])
            )
            
            # Update portfolio state
            portfolio.step(new_hidden_states)