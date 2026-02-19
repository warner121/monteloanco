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
        logits = torch.log(probs.clamp(min=1e-8))
        return logits.masked_fill(Template.MASK.to(logits.device), float("-inf"))

    @staticmethod
    def apply_mask(logits: torch.Tensor) -> torch.Tensor:
        return logits.masked_fill(Template.MASK.to(logits.device), float("-inf"))

    @staticmethod
    def batch_logits(logits: torch.Tensor, batch_size: int) -> torch.Tensor:
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
        ), None


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
            return logits.unsqueeze(0).expand(batch_size, -1, -1), None
        return logits, None


class LearnedTransition(TransitionMatrixProvider):
    """
    Learned transition matrix with optional per-loan heterogeneity.

    Two uncertainty modes (set via `uncertainty`):

    'none'  — identical to the original deterministic version.
              One shared param per batch_id, no per-loan variation.

    'loan'  — deterministic per-loan logit offsets stored as a pyro.param
              (total_size, 64).  No stochastic latents → clean gradients,
              but each loan gets its own adjustment around the shared baseline.
              Requires total_size to be set.

    In both modes the observation-level std is a learned scalar param
    (obs_log_std), so the Normal likelihoods can adapt their noise level
    during training — this is where the primary "uncertainty" lives.
    """

    def __init__(self, name, init_logits=None, trainable=True,
                 uncertainty='loan', total_size=None, prior_strength=1.0):
        """
        Args:
            name:           Unique name for pyro params.
            init_logits:    (8,8) initial logits (masked inf values are reset to 0).
            trainable:      If False, all params are detached (frozen).
            uncertainty:    'none' or 'loan'  (see class docstring).
            total_size:     Total number of loans in the dataset.
                            Required when uncertainty='loan'.
            prior_strength: Scale of the L2 regularisation on per-loan offsets.
                            Larger values pull short-history loans harder toward
                            the shared baseline.  Tune in [0.1, 10.0].
                            Has no effect when uncertainty='none'.
        """
        assert uncertainty in ('none', 'loan'), \
            "uncertainty must be 'none' or 'loan'"
        self.name = name
        self.trainable = trainable
        self.uncertainty = uncertainty
        self.total_size = total_size
        self.prior_strength = prior_strength

        if init_logits is not None:
            init = init_logits.clone()
            init[torch.isinf(init)] = 0.0
            self.init_logits_unmasked = init
        else:
            self.init_logits_unmasked = None

    def get_logits(self, batch_id, batch_idx, batch_size, device):
        if self.init_logits_unmasked is None:
            raise ValueError("init_logits must be provided for LearnedTransition")
        if self.total_size is None:
            raise ValueError("total_size must be set for LearnedTransition")

        flat_init = self.init_logits_unmasked.flatten().to(device)

        # ── Shared baseline: fixed prior, not trainable. ───────────────────────
        # init_logits are broadcast to all loans but held constant — the only
        # trainable signal comes from per-loan offsets below.
        baseline_all = flat_init.unsqueeze(0).expand(self.total_size, -1)  # (total_size, 64)

        logits_flat = baseline_all[batch_idx]   # (batch_size, 64) always correct

        # ── Per-loan deterministic offsets (optional) ──────────────────────────
        if self.uncertainty == 'loan':
            offsets_all = pyro.param(
                f"{self.name}_loan_offsets",
                torch.zeros(self.total_size, flat_init.shape[0], device=device)
            )
            if not self.trainable:
                offsets_all = offsets_all.detach()

            batch_offsets = offsets_all[batch_idx]   # (batch_size, 64)
            logits_flat = logits_flat + batch_offsets
        else:
            batch_offsets = None

        tmat = logits_flat.view(-1, 8, 8)
        return Template.apply_mask(tmat), batch_offsets


class Portfolio:
    """Manages batch of loans with vectorized operations"""

    def __init__(self, loan_amnt, installments, int_rate,
                 num_timesteps=60, total_pre_chargeoff=None,
                 last_pymnt_amnt=None, device='cuda:0', scaling_factor=1_000_000):
        self.batch_size = len(loan_amnt)
        self.loan_amnt = loan_amnt / scaling_factor
        self.installments = installments / scaling_factor
        self.int_rate = int_rate
        self.device = device
        self.scaling_factor = scaling_factor

        if torch.is_tensor(num_timesteps):
            self.num_timesteps = num_timesteps.to(device)
        else:
            self.num_timesteps = torch.full((self.batch_size,), num_timesteps,
                                            dtype=torch.long, device=device)

        self.max_timesteps = self.num_timesteps.max().item()

        self.total_pre_chargeoff = (total_pre_chargeoff / scaling_factor
                                    if torch.is_tensor(total_pre_chargeoff) else None)
        self.last_pymnt_amnt = (last_pymnt_amnt / scaling_factor
                                if torch.is_tensor(last_pymnt_amnt) else None)

        self.current_balances = self.loan_amnt.clone()
        self.current_interest_owed = torch.zeros(self.batch_size, device=device)
        self.current_hidden_states = torch.ones(self.batch_size, dtype=torch.int32, device=device)

        self.balances_history = [self.current_balances.clone()]
        self.interest_paid_history = [torch.zeros(self.batch_size, device=device)]
        self.principal_paid_history = [torch.zeros(self.batch_size, device=device)]
        self.payments_history = [torch.zeros(self.batch_size, device=device)]
        self.hidden_states_history = [self.current_hidden_states.clone()]

    def _apply_timestep_mask(self, target):
        timestep_range = torch.arange(1, self.max_timesteps + 1,
                                      device=self.device).unsqueeze(1)
        mask = timestep_range <= self.num_timesteps.unsqueeze(0)
        return target * mask

    def calculate_payment(self, new_hidden_states, old_hidden_states):
        payment = torch.where(
            new_hidden_states < 7,
            (old_hidden_states - new_hidden_states + 1) * self.installments,
            torch.zeros(self.batch_size, device=self.device)
        )
        payment = torch.where(
            new_hidden_states == 0,
            self.current_balances + self.current_interest_owed,
            payment
        )
        return payment

    def apply_payment(self, payment):
        interest_payment = torch.minimum(payment, self.current_interest_owed)
        principal_payment = torch.clamp(payment - interest_payment, min=0)
        new_balance = torch.clamp(self.current_balances - principal_payment, min=0)
        return interest_payment, principal_payment, new_balance

    def accrue_interest(self):
        self.current_interest_owed += self.current_balances * self.int_rate / 1200

    def step(self, new_hidden_states):
        self.accrue_interest()
        payment = self.calculate_payment(new_hidden_states, self.current_hidden_states)
        interest_paid, principal_paid, new_balance = self.apply_payment(payment)
        self.current_balances = new_balance
        self.current_interest_owed = self.current_interest_owed - interest_paid
        self.current_hidden_states = new_hidden_states
        self.balances_history.append(new_balance.clone())
        self.interest_paid_history.append(interest_paid)
        self.principal_paid_history.append(principal_paid)
        self.payments_history.append(payment)
        self.hidden_states_history.append(new_hidden_states.clone())

    def get_histories(self):
        return {
            'hidden_states': torch.stack(self.hidden_states_history[1:]),
            'payments': torch.stack(self.payments_history[1:]) * self.scaling_factor,
            'interest_paid': torch.stack(self.interest_paid_history[1:]) * self.scaling_factor,
            'principal_paid': torch.stack(self.principal_paid_history[1:]) * self.scaling_factor
        }

    def get_total_pre_chargeoff(self):
        payments = torch.stack(self.payments_history[1:])
        return self._apply_timestep_mask(payments).sum(0)

    def get_last_payment(self):
        payments = torch.stack(self.payments_history[1:])
        return self._apply_timestep_mask(payments).max(0)[0]


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

    tmat_logits, loan_offsets = tmat_provider.get_logits(batch_id, batch_idx, batch_size, device)

    # ── Learned observation noise: one scalar std per dataset, trained via SVI ──
    # log-parameterised so it's always positive; init ~50 / scaling_factor
    obs_std = pyro.param(
        f"obs_log_std_{batch_id}",
        torch.tensor((50. / scaling_factor)).log().to(device)
    ).exp()

    with pyro.plate(f"batch_{batch_id}", batch_size, dim=-1):

        # ── Exposure-weighted L2 prior on per-loan offsets ─────────────────────
        # Penalty = -prior_strength * sum(offsets^2) / T_i
        # Short loans (low T) → large penalty → pulled toward shared baseline.
        # Long loans (high T) → small penalty → free to specialise.
        if loan_offsets is not None:
            inv_T = 1.0 / portfolio.num_timesteps.float().clamp(min=1)  # (batch,)
            l2 = (loan_offsets ** 2).sum(-1)                            # (batch,)
            pyro.factor(
                f"offset_prior_{batch_id}",
                -tmat_provider.prior_strength * inv_T * l2
            )

        for t in range(1, portfolio.max_timesteps + 1):
            new_hidden_states = pyro.sample(
                f"h_{batch_id}_{t}",
                dist.Categorical(logits=tmat_logits[
                    torch.arange(batch_size, device=device),
                    portfolio.current_hidden_states
                ])
            )
            portfolio.step(new_hidden_states)

        if torch.is_tensor(total_pre_chargeoff):
            pred = portfolio.get_total_pre_chargeoff()
            # Scale std by sqrt(T) so longer loans get proportionally wider likelihood
            std = obs_std * torch.sqrt(portfolio.num_timesteps.float())
            pyro.sample(
                f"obs_total_{batch_id}",
                dist.Normal(pred, std),
                obs=total_pre_chargeoff / scaling_factor
            )

        if torch.is_tensor(last_pymnt_amnt):
            pred = portfolio.get_last_payment()
            std = obs_std * torch.sqrt(portfolio.num_timesteps.float())
            pyro.sample(
                f"obs_last_{batch_id}",
                dist.Normal(pred, std),
                obs=last_pymnt_amnt / scaling_factor
            )

    return portfolio


# Guide is identical to deterministic original — no stochastic latents to approximate.
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

    portfolio = Portfolio(
        loan_amnt, installments, int_rate,
        num_timesteps, total_pre_chargeoff, last_pymnt_amnt,
        device, scaling_factor
    )

    tmat_logits, _ = tmat_provider.get_logits(batch_id, batch_idx, batch_size, device)

    with pyro.plate(f"batch_{batch_id}", batch_size, dim=-1):
        for t in range(1, portfolio.max_timesteps + 1):
            new_hidden_states = pyro.sample(
                f"h_{batch_id}_{t}",
                dist.Categorical(logits=tmat_logits[
                    torch.arange(batch_size, device=device),
                    portfolio.current_hidden_states
                ])
            )
            portfolio.step(new_hidden_states)
