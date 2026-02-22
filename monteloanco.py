import torch
import pyro
import pyro.distributions as dist


class Template:
    """
    Structural constants and utility operations for the 8-state Markov chain.

    States 0–7 encode loan performance buckets (e.g. prepaid, current, DPD bands,
    default).  The MASK encodes structurally forbidden transitions — entries set to
    True are clamped to -inf in logit space so the Categorical sampler assigns them
    zero probability.  DEMO_PROBS is the portfolio-level prior transition matrix,
    elicited from domain knowledge; it is now the fixed baseline — no learned global
    offset layer sits above it.
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

# Pre-compute the log-probability form of the prior transition matrix.
# This is now the fixed baseline used directly in model/guide — it is never
# updated by gradient descent.
Template.DEMO_LOGITS = Template.probs_to_logits(Template.DEMO_PROBS)


# ── Parameter helpers ─────────────────────────────────────────────────────────
#
# The hierarchy has been reduced to two layers:
#
#   θ_i  =  mask(DEMO_LOGITS + α_i)
#
# DEMO_LOGITS is fixed (domain-knowledge baseline).  α_i is the only stochastic
# layer — a per-loan offset drawn from a Normal prior.  There is no learned
# global matrix sitting between them.
#
# Call order within a training step:
#
#   sigma_obs              = register_global_params(...)   # outside plate
#   alpha_loc, alpha_scale = register_loan_params(...)     # outside plate
#   with pyro.plate(...):
#       loan_offsets = pyro.sample("alpha_...", ...)       # inside plate
#       tmat_logits  = build_tmat_logits(...)              # inside plate
#       ...


def register_global_params(batch_id: str, device: str, scaling_factor: float):
    """
    Register and return the scalar observation-noise parameter.

    The learned global logit matrix has been removed.  DEMO_LOGITS is the fixed
    baseline; only σ_obs remains as a trained scalar.

    Parameters
    ----------
    batch_id       : string prefix for all Pyro site names in this batch.
    device         : torch device string.
    scaling_factor : monetary divisor; initialises sigma_obs at a realistic value.

    Returns
    -------
    sigma_obs : scalar — homoskedastic observation noise.  Positive-constrained.
    """
    sigma_obs = pyro.param(
        f"{batch_id}_sigma_obs",
        torch.tensor(50. / scaling_factor, device=device),
        constraint=dist.constraints.positive,
    )   # scalar

    return sigma_obs


def register_loan_params(batch_id: str, batch_idx: torch.Tensor,
                         total_size: int, device: str):
    """
    Register the full-dataset variational parameter tables and return the rows
    corresponding to the current batch.

    The tables are indexed by loan position in the full dataset.  Only the rows
    for batch_idx are returned; gradients flow back to those rows only.

    Parameters
    ----------
    batch_id   : string prefix for Pyro site names.
    batch_idx  : 1-D integer tensor of loan indices into the full dataset,
                 shape (batch_size,).
    total_size : total number of distinct loans in the dataset.
    device     : torch device string.

    Returns
    -------
    alpha_loc   : (batch_size, 8, 8) — variational mean for each loan's offset α_i.
                  Initialised at zero (prior mean).
    alpha_scale : (batch_size, 8, 8) — variational std for each loan's offset α_i.
                  Positive-constrained; initialised small for numerical stability.
    """
    alpha_loc_all = pyro.param(
        f"{batch_id}_alpha_loc",
        torch.zeros(total_size, 8, 8, device=device),
    )   # (total_size, 8, 8)

    alpha_scale_all = pyro.param(
        f"{batch_id}_alpha_scale",
        torch.full((total_size, 8, 8), 0.1, device=device),
        constraint=dist.constraints.positive,
    )   # (total_size, 8, 8)

    return alpha_loc_all[batch_idx], alpha_scale_all[batch_idx]


def build_tmat_logits(loan_offsets: torch.Tensor, device: str):
    """
    Construct per-loan transition logits from the fixed DEMO_LOGITS baseline and
    sampled per-loan offsets.  Pure tensor arithmetic — no Pyro calls.

    Parameters
    ----------
    loan_offsets : (batch_size, 8, 8) — sampled or posterior-mean per-loan
                   offsets α_i.
    device       : torch device string (used to move DEMO_LOGITS if needed).

    Returns
    -------
    tmat_logits : (batch_size, 8, 8) — structurally masked per-loan logits,
                  ready for Categorical sampling or softmax inspection.
    """
    baseline = Template.DEMO_LOGITS.to(device)   # (8, 8), fixed
    return Template.apply_mask(
        baseline.unsqueeze(0) + loan_offsets
    )   # (batch_size, 8, 8)


# ── Cashflow simulator ────────────────────────────────────────────────────────

class Portfolio:
    """
    Vectorised cashflow simulator for a batch of loans evolving under a
    discrete-time Markov chain over the 8-state delinquency lattice.

    At each timestep the hidden state determines the payment fraction collected;
    the Portfolio object accumulates payment histories for downstream likelihood
    evaluation.
    """

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


# ── Model and guide ───────────────────────────────────────────────────────────
#
# The hierarchy is now:
#
#   θ_i  =  mask(DEMO_LOGITS + α_i)
#
# DEMO_LOGITS is the fixed domain-knowledge baseline (never trained).
# α_i is the sole stochastic layer — per-loan offsets from that baseline.
# There is no global logit matrix and no σ_global parameter.
#
# Both functions have identical signatures.  Structure in each:
#
#   1. register_global_params()  — σ_obs only              (outside plate)
#   2. register_loan_params()    — alpha_loc, alpha_scale   (outside plate)
#   3. pyro.plate:
#        a. pyro.sample alpha    — loan offsets α_i         (inside plate)
#        b. build_tmat_logits()  — pure tensor op           (inside plate)
#        c. Markov chain loop                                (inside plate)
#        d. observation sites                                (inside plate)

def model(
    batch_id,
    batch_idx,
    installments,
    loan_amnt,
    int_rate,
    total_size,
    total_pre_chargeoff=None,
    last_pymnt_amnt=None,
    num_timesteps=60,
    device="cuda:0",
    scaling_factor=1_000_000,
):
    """
    Generative model for a batch of loans.

    Per-loan transition logits are:
        θ_i  =  mask(DEMO_LOGITS + α_i)

    DEMO_LOGITS is the fixed domain-knowledge prior matrix.  α_i is a stochastic
    per-loan offset drawn from an exposure-scaled Normal prior:

        α_i  ~  Normal(0, 1 / sqrt(T_i))

    where T_i is the number of observed timesteps for loan i (clamped to 1).
    Loans with short histories have wide priors and stay close to DEMO_LOGITS;
    loans with long histories are effectively pinned by their own data.

    Observation likelihood (when targets are provided):
        total_pre_chargeoff  ~  Normal(simulated_total,  σ_obs)
        last_pymnt_amnt      ~  Normal(simulated_last,   σ_obs)
    """
    batch_size = len(batch_idx)

    portfolio = Portfolio(
        loan_amnt, installments, int_rate,
        num_timesteps, total_pre_chargeoff, last_pymnt_amnt,
        device, scaling_factor,
    )

    # ── Global params (outside plate) — σ_obs only ───────────────────────────
    sigma_obs = register_global_params(batch_id, device, scaling_factor)

    # Exposure-scaled prior std: wide for short histories, narrow for long.
    T = portfolio.num_timesteps.float().clamp(min=1)                   # (batch_size,)
    prior_std = (1.0 / T.sqrt()).view(-1, 1, 1).expand(batch_size, 8, 8)

    # ── Loan-level params (outside plate) ────────────────────────────────────
    alpha_loc, alpha_scale = register_loan_params(batch_id, batch_idx, total_size, device)

    with pyro.plate(f"batch_{batch_id}", batch_size, dim=-1):

        # α_i ~ Normal(0, 1 / sqrt(T_i))
        # to_event(2) declares (8, 8) as the event shape; the plate dim is loans.
        loan_offsets = pyro.sample(
            f"alpha_{batch_id}",
            dist.Normal(
                torch.zeros(batch_size, 8, 8, device=device),
                prior_std,
            ).to_event(2)
        )   # (batch_size, 8, 8)

        tmat_logits = build_tmat_logits(loan_offsets, device)

        for t in range(1, portfolio.max_timesteps + 1):
            new_hidden_states = pyro.sample(
                f"h_{batch_id}_{t}",
                dist.Categorical(logits=tmat_logits[
                    torch.arange(batch_size, device=device),
                    portfolio.current_hidden_states,
                ])
            )
            portfolio.step(new_hidden_states)

        if torch.is_tensor(total_pre_chargeoff):
            pred = portfolio.get_total_pre_chargeoff()
            pyro.sample(
                f"obs_total_{batch_id}",
                dist.Normal(pred, sigma_obs),
                obs=total_pre_chargeoff / scaling_factor,
            )

        if torch.is_tensor(last_pymnt_amnt):
            pred = portfolio.get_last_payment()
            pyro.sample(
                f"obs_last_{batch_id}",
                dist.Normal(pred, sigma_obs),
                obs=last_pymnt_amnt / scaling_factor,
            )

    return portfolio


def guide(
    batch_id,
    batch_idx,
    installments,
    loan_amnt,
    int_rate,
    total_size,
    total_pre_chargeoff=None,
    last_pymnt_amnt=None,
    num_timesteps=60,
    device="cuda:0",
    scaling_factor=1_000_000,
):
    """
    Variational guide for model().

    The only deterministic parameter is σ_obs (registered via pyro.param, no
    sampling).  The guide for α_i is:

        q(α_i)  =  Normal(loc_i, scale_i)

    where loc_i and scale_i are free (8×8) variational parameters stored in
    full-dataset tables (total_size, 8, 8) and indexed by batch_idx.
    """
    batch_size = len(batch_idx)

    portfolio = Portfolio(
        loan_amnt, installments, int_rate,
        num_timesteps, total_pre_chargeoff, last_pymnt_amnt,
        device, scaling_factor,
    )

    # ── Global params (outside plate) — σ_obs only ───────────────────────────
    register_global_params(batch_id, device, scaling_factor)

    # ── Loan-level variational params (outside plate) ────────────────────────
    alpha_loc, alpha_scale = register_loan_params(batch_id, batch_idx, total_size, device)

    with pyro.plate(f"batch_{batch_id}", batch_size, dim=-1):

        # q(α_i) = Normal(loc_i, scale_i)
        loan_offsets = pyro.sample(
            f"alpha_{batch_id}",
            dist.Normal(alpha_loc, alpha_scale).to_event(2)
        )   # (batch_size, 8, 8)

        tmat_logits = build_tmat_logits(loan_offsets, device)

        for t in range(1, portfolio.max_timesteps + 1):
            new_hidden_states = pyro.sample(
                f"h_{batch_id}_{t}",
                dist.Categorical(logits=tmat_logits[
                    torch.arange(batch_size, device=device),
                    portfolio.current_hidden_states,
                ])
            )
            portfolio.step(new_hidden_states)
