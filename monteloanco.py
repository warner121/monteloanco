import torch
import pyro
import pyro.distributions as dist


class Template:
    """
    Structural constants and domain-knowledge prior for the 8-state delinquency
    Markov chain.

    States 0-7 encode loan performance buckets (prepaid, current, DPD-30, DPD-60,
    DPD-90, DPD-120, DPD-150+, charged-off).

    MASK
        Boolean (8, 8) tensor encoding structurally forbidden transitions -- e.g. a
        loan cannot jump from current directly to DPD-90 in one period.  Forbidden
        entries are clamped to -inf in logit space, ensuring the Categorical
        likelihood assigns them exactly zero probability regardless of the sampled
        logit perturbations alpha_i.

    DEMO_PROBS / DEMO_LOGITS
        The fixed prior mean for transition probabilities, elicited from domain
        knowledge.  DEMO_LOGITS = log(DEMO_PROBS) after masking, and serves as the
        baseline around which per-loan logit perturbations alpha_i are centred.
        Neither tensor is a learnable parameter -- they are fixed constants that
        anchor the generative model to expert belief.
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
        """Convert a probability matrix to log-space, then apply the structural mask."""
        logits = torch.log(probs.clamp(min=1e-8))
        return logits.masked_fill(Template.MASK.to(logits.device), float("-inf"))

    @staticmethod
    def apply_mask(logits: torch.Tensor) -> torch.Tensor:
        """Zero out structurally forbidden transitions in logit space (-inf)."""
        return logits.masked_fill(Template.MASK.to(logits.device), float("-inf"))


# Fixed prior mean for per-loan transition logits, derived from domain knowledge.
# DEMO_LOGITS is the baseline around which per-loan logit perturbations alpha_i
# are centred.  It is never modified during inference.
Template.DEMO_LOGITS = Template.probs_to_logits(Template.DEMO_PROBS)


def construct_transition_logits(per_loan_logit_offsets: torch.Tensor, device: str):
    """
    Construct per-loan transition logit matrices as:

        theta_i  =  mask( DEMO_LOGITS + alpha_i )

    where DEMO_LOGITS is the fixed (8, 8) domain-knowledge baseline (prior mean
    for transition logits) and alpha_i are the sampled per-loan logit perturbations.
    This is a deterministic operation -- no random variables are introduced here.

    Parameters
    ----------
    per_loan_logit_offsets : (batch_size, 8, 8)
        Per-loan logit perturbations alpha_i sampled from Normal(0, sigma_i).
    device : str
        Torch device string; DEMO_LOGITS is moved here if necessary.

    Returns
    -------
    transition_logits : (batch_size, 8, 8)
        Structurally masked per-loan transition logits theta_i, ready to
        parameterise a Categorical distribution over next delinquency states.
    """
    # DEMO_LOGITS: fixed (8, 8) prior mean for transition logits -- never trained.
    prior_mean_logits = Template.DEMO_LOGITS.to(device)
    return Template.apply_mask(
        prior_mean_logits.unsqueeze(0) + per_loan_logit_offsets
    )   # (batch_size, 8, 8)


# -- Cash-flow simulator ------------------------------------------------------

class Portfolio:
    """
    Vectorised cash-flow simulator for a batch of loans.

    Conditional on a sequence of latent delinquency states {h_{i,t}}, this class
    deterministically computes payment trajectories and summary observables.  It
    forms the likelihood bridge between the latent Markov chain and observed loan
    performance data.

    Role in the generative model
    ----------------------------
    Given per-loan transition logits theta_i (a deterministic function of the
    latent logit perturbation alpha_i), the simulator propagates each loan through
    its delinquency states and accumulates:

      * hidden_states_history  -- latent delinquency-state sequence
      * payments_history       -- payment received at each period (deterministic
                                  conditional on states and contract terms)
      * summary observables used in the likelihood:
          - get_total_pre_chargeoff() -> compared to observed total_pre_chargeoff
          - get_last_payment()        -> compared to observed last_pymnt_amnt
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

        # T_i: observed history length for each loan -- the number of latent states
        # in the delinquency-state sequence and the simulation horizon per loan.
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
        # Latent delinquency states: initialised to state 1 (current) for all loans.
        self.current_hidden_states = torch.ones(self.batch_size, dtype=torch.int32, device=device)

        self.balances_history = [self.current_balances.clone()]
        self.interest_paid_history = [torch.zeros(self.batch_size, device=device)]
        self.principal_paid_history = [torch.zeros(self.batch_size, device=device)]
        self.payments_history = [torch.zeros(self.batch_size, device=device)]
        self.hidden_states_history = [self.current_hidden_states.clone()]

    def _apply_timestep_mask(self, target):
        """Zero out entries beyond each loan's observed history length T_i."""
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
        """Advance all loans by one period given sampled latent delinquency states h_t."""
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
        """Simulated total payments within the observed window T_i (observed summary statistic)."""
        payments = torch.stack(self.payments_history[1:])
        return self._apply_timestep_mask(payments).sum(0)

    def get_last_payment(self):
        """Simulated last payment amount within the observed window (observed summary statistic)."""
        payments = torch.stack(self.payments_history[1:])
        return self._apply_timestep_mask(payments).max(0)[0]


# -- Model and guide ----------------------------------------------------------
#
# Generative hierarchy
# --------------------
# Level 0 -- global latent (shared across all loans, sampled outside the plate):
#
#   sigma_base  ~  LogNormal(0, 1)
#
#     The global prior scale for per-loan logit perturbations alpha_i.  Sampled
#     rather than point-estimated, so posterior uncertainty about the overall
#     magnitude of deviation from the domain-knowledge baseline DEMO_LOGITS is
#     propagated through inference.
#
# Level 1 -- per-loan prior std (deterministic function of observed history T_i,
#            computed outside the plate):
#
#   sigma_i  =  sigma_base * sqrt( T_i / (T_i + T0) )
#
#     History-adjusted prior std for loan i.  T_i = num_timesteps[i] is the
#     number of observed periods; T0 = 12 months is a fixed hyperparameter.
#     Loans with short histories (T_i << T0) have sigma_i ~ 0, shrinking alpha_i
#     tightly toward DEMO_LOGITS.  Loans with rich histories (T_i >> T0) have
#     sigma_i ~ sigma_base, allowing full deviation from the baseline.
#
# Level 2 -- per-loan logit perturbations (latent, sampled inside the plate):
#
#   alpha_i  ~  Normal(0, sigma_i)    shape: (8, 8), declared via .to_event(2)
#   theta_i   =  mask( DEMO_LOGITS + alpha_i )
#
#     alpha_i is the per-loan logit perturbation matrix -- the sole stochastic
#     offset from the fixed domain-knowledge baseline DEMO_LOGITS.  theta_i are
#     the resulting per-loan transition logits, structurally masked.
#
# Level 3 -- latent delinquency-state sequence (sampled inside the plate and loop):
#
#   h_{i,t}  ~  Categorical( softmax(theta_i[h_{i,t-1}, :]) )    t = 1 ... T_i
#
#     The hidden Markov chain.  State evolution is Markovian and conditionally
#     independent across loans given their respective theta_i.  The cash-flow
#     simulator is deterministic conditional on {h_{i,t}}.
#
# Level 4 -- observations (likelihood, conditioned on simulated trajectories):
#
#   total_pre_chargeoff_i  ~  Normal( sim_total_i,  sigma_obs_i )   [if observed]
#   last_pymnt_amnt_i      ~  Normal( sim_last_i,   sigma_obs_i )   [if observed]
#
#     sim_total_i and sim_last_i are deterministic summaries of the payment
#     trajectory generated by the simulator conditional on {h_{i,t}}.
#     sigma_obs_i = (50 / scaling_factor) * sqrt(T_i) scales observation noise
#     with history length.


def model(
    batch_id,
    batch_idx,
    installments,
    loan_amnt,
    int_rate,
    total_pre_chargeoff=None,
    last_pymnt_amnt=None,
    num_timesteps=60,
    device="cuda:0",
    scaling_factor=1_000_000,
):
    """
    Generative model for a batch of loans under a hierarchical Bayesian
    hidden Markov chain.

    Generative story
    ----------------
    1. Global prior scale for logit perturbations (outside the loan plate):

           sigma_base  ~  LogNormal(0, 1)

       sigma_base is the global prior scale governing how far per-loan transition
       logits may deviate from the domain-knowledge baseline DEMO_LOGITS.  It is
       a latent variable -- not a point-estimated parameter -- so its posterior
       uncertainty propagates to all per-loan quantities.

    2. History-adjusted prior std per loan (deterministic, outside the plate):

           sigma_i  =  sigma_base * sqrt( T_i / (T_i + T0) )

       T_i = simulator.num_timesteps[i] is the observed history length for loan
       i; T0 = 12 months is a fixed hyperparameter.  sigma_i is reshaped to
       (batch_size, 1, 1) to broadcast across the (8, 8) logit event shape.

    3. Per-loan logit perturbations and transition logits (inside the loan plate):

           alpha_i  ~  Normal(0, sigma_i),   shape (8, 8) via .to_event(2)
           theta_i   =  mask( DEMO_LOGITS + alpha_i )

       alpha_i is the per-loan logit perturbation matrix sampled from a zero-mean
       Normal with history-adjusted prior std sigma_i.  theta_i are the resulting
       per-loan transition logits, with structurally forbidden entries masked.

    4. Latent delinquency-state sequence (inside the plate, inside the loop):

           h_{i,t}  ~  Categorical( softmax(theta_i[h_{i,t-1}, :]) )

       The hidden Markov chain evolves conditional on theta_i.  State evolution is
       Markovian and conditionally independent across loans given their theta_i.
       The cash-flow simulator is deterministic conditional on {h_{i,t}}.

    5. Observed summary statistics -- likelihood (inside the plate):

           total_pre_chargeoff_i  ~  Normal( sim_total_i,  sigma_obs_i )
           last_pymnt_amnt_i      ~  Normal( sim_last_i,   sigma_obs_i )

       sim_total_i and sim_last_i are deterministic summaries of the payment
       trajectory generated by the simulator conditional on {h_{i,t}}.
       sigma_obs_i = (50 / scaling_factor) * sqrt(T_i) scales observation noise
       with history length.  These sites are omitted when targets are not
       provided (e.g. during prior predictive sampling).
    """
    batch_size = len(batch_idx)

    # Instantiate the cash-flow simulator.  It accumulates simulated payment
    # trajectories conditional on the latent delinquency-state sequence; its
    # summary statistics form the observation targets for the likelihood.
    simulator = Portfolio(
        loan_amnt, installments, int_rate,
        num_timesteps, total_pre_chargeoff, last_pymnt_amnt,
        device, scaling_factor,
    )

    # -- Level 0: global prior scale for logit perturbations ------------------
    # sigma_base ~ LogNormal(0, 1)
    # Sampled outside the loan plate -- sigma_base is shared across all loans.
    # LogNormal ensures sigma_base > 0 without an explicit positivity constraint.
    global_logit_scale = pyro.sample(
        "sigma_base",
        dist.LogNormal(torch.tensor(0.0, device=device),
                       torch.tensor(1.0, device=device)),
    )

    # -- Level 1: history-adjusted prior std for each loan --------------------
    # sigma_i = sigma_base * sqrt( T_i / (T_i + T0) )
    #
    # T0 = 12 months (fixed hyperparameter) sets the crossover scale:
    #   T_i << T0  ->  sigma_i ~ 0         (alpha_i shrunk to DEMO_LOGITS)
    #   T_i >> T0  ->  sigma_i ~ sigma_base (alpha_i free to deviate)
    #
    # Deterministic function of observed data -- no Pyro site.
    T0 = 12.0   # reference horizon (months); fixed hyperparameter
    history_lengths = simulator.num_timesteps.float()                  # T_i, (batch_size,)
    per_loan_prior_std = (
        global_logit_scale
        * torch.sqrt(history_lengths / (history_lengths + T0))
    )                                                                   # (batch_size,)
    per_loan_prior_std = per_loan_prior_std.reshape(batch_size, 1, 1)  # broadcast over (8, 8)

    with pyro.plate(f"batch_{batch_id}", batch_size, dim=-1):

        # -- Level 2: per-loan logit perturbations alpha_i --------------------
        # alpha_i ~ Normal(0, sigma_i),  shape (8, 8) declared via .to_event(2).
        # The plate dimension (-1) indexes loans; (8, 8) is the event shape.
        # Prior mean is zero: no expected deviation from the domain-knowledge
        # baseline DEMO_LOGITS without evidence from observed data.
        per_loan_logit_offsets = pyro.sample(
            f"alpha_{batch_id}",
            dist.Normal(
                torch.zeros(batch_size, 8, 8, device=device),  # prior mean: zero deviation
                per_loan_prior_std,                             # prior std: history-shrunk
            ).to_event(2)
        )   # (batch_size, 8, 8)

        # Per-loan transition logits: theta_i = mask( DEMO_LOGITS + alpha_i ).
        # Deterministic given alpha_i -- no new random variables introduced.
        per_loan_transition_logits = construct_transition_logits(
            per_loan_logit_offsets, device
        )

        # -- Level 3: latent delinquency-state sequence -----------------------
        # h_{i,t} ~ Categorical( softmax(theta_i[h_{i,t-1}, :]) )
        # State evolution is Markovian and conditionally independent across loans
        # given theta_i.  The cash-flow simulator is deterministic conditional on
        # the sampled state sequence.
        for t in range(1, simulator.max_timesteps + 1):
            next_delinquency_states = pyro.sample(
                f"h_{batch_id}_{t}",
                dist.Categorical(logits=per_loan_transition_logits[
                    torch.arange(batch_size, device=device),
                    simulator.current_hidden_states,
                ])
            )
            # Advance the simulator one period conditional on the sampled latent
            # states.  Payment trajectories are deterministic given {h_{i,t}}.
            simulator.step(next_delinquency_states)

        # -- Level 4: observed summary statistics (likelihood) ----------------
        # The cash-flow simulator is deterministic conditional on {h_{i,t}};
        # the two observation sites below close the likelihood by comparing
        # simulated summaries to their observed counterparts.
        #
        # Observation noise sigma_obs_i = (50 / scaling_factor) * sqrt(T_i)
        # scales with history length to reflect growing absolute payment variance.

        if torch.is_tensor(total_pre_chargeoff):
            # Observation: total payments received within the observed window.
            simulated_total = simulator.get_total_pre_chargeoff()
            obs_std = (50. / scaling_factor) * torch.sqrt(simulator.num_timesteps.float())
            pyro.sample(
                f"obs_total_{batch_id}",
                dist.Normal(simulated_total, obs_std),
                obs=total_pre_chargeoff / scaling_factor
            )

        if torch.is_tensor(last_pymnt_amnt):
            # Observation: last payment amount within the observed window.
            simulated_last = simulator.get_last_payment()
            obs_std = (50. / scaling_factor) * torch.sqrt(simulator.num_timesteps.float())
            pyro.sample(
                f"obs_last_{batch_id}",
                dist.Normal(simulated_last, obs_std),
                obs=last_pymnt_amnt / scaling_factor
            )

    return simulator


def guide(
    batch_id,
    batch_idx,
    installments,
    loan_amnt,
    int_rate,
    total_pre_chargeoff=None,
    last_pymnt_amnt=None,
    num_timesteps=60,
    device="cuda:0",
    scaling_factor=1_000_000,
):
    """
    Mean-field variational guide for model().

    Defines a factored approximation to the joint posterior over the global
    prior scale and all per-loan logit perturbations:

        q(sigma_base, {alpha_i})  =  q(sigma_base)  *  prod_i  q(alpha_i)

    All site names match the model exactly so the ELBO can pair model and guide
    distributions correctly.

    Global prior scale  (outside the loan plate)
    ---------------------------------------------
        q(sigma_base)  =  LogNormal(sigma_base_loc, sigma_base_scale)

    sigma_base_loc   (unconstrained scalar, init 0.0) is the log-space
    variational mean; sigma_base_scale (positive scalar, init 0.5) is the
    log-space variational std.  LogNormal matches the model prior's support
    (R+) and provides a flexible unimodal approximation.  Sampled outside the
    loan plate, mirroring the model's generative structure.

    Per-loan logit perturbations  (inside the loan plate)
    ------------------------------------------------------
        q(alpha_i)  =  Normal(mu_i, s_i)

    mu_i is a free (8, 8) variational mean for the per-loan logit perturbation
    matrix; s_i is a per-loan scalar variational std that broadcasts over the
    (8, 8) event shape.  Both are stored in per-batch parameter tables keyed by
    batch_id; batch_idx selects the relevant rows -- arange(batch_size) during
    training, specific loan indices at inference time.

    Note: the per-loan prior std sigma_i (a deterministic function of sigma_base
    and T_i, computed in the model) affects only the KL term in the ELBO.  The
    guide family for alpha_i is independent of sigma_i and remains a simple Normal.
    """
    batch_size = len(batch_idx)

    # The simulator is instantiated here solely to access num_timesteps (= T_i)
    # and max_timesteps for each loan in the batch.  Its payment trajectory is
    # not used probabilistically in the guide -- only the latent sites
    # alpha_{batch_id} and h_{batch_id}_{t} are sampled.
    simulator = Portfolio(
        loan_amnt, installments, int_rate,
        num_timesteps, total_pre_chargeoff, last_pymnt_amnt,
        device, scaling_factor,
    )

    # -- Variational distribution for the global prior scale ------------------
    # q(sigma_base) = LogNormal(sigma_base_loc, sigma_base_scale)
    # Sampled outside the loan plate, matching the model's generative structure.
    #
    # sigma_base_loc  : log-space variational mean for sigma_base.
    # sigma_base_scale: log-space variational std for sigma_base (positive).
    variational_logscale_loc = pyro.param(
        "sigma_base_loc",
        torch.tensor(0.0, device=device),
    )
    variational_logscale_std = pyro.param(
        "sigma_base_scale",
        torch.tensor(0.5, device=device),
        constraint=dist.constraints.positive,
    )
    pyro.sample(
        "sigma_base",
        dist.LogNormal(variational_logscale_loc, variational_logscale_std),
    )

    # -- Variational distribution for per-loan logit perturbations alpha_i ----
    # q(alpha_i) = Normal(mu_i, s_i)
    #
    # Parameter tables are sized to the full batch at registration time;
    # batch_idx selects the rows for the current mini-batch or inference query.
    #
    # alpha_variational_loc   : (batch_size, 8, 8) -- variational mean mu_i;
    #                           one free parameter per transition logit per loan.
    # alpha_variational_scale : (batch_size, 1, 1) -- per-loan variational std s_i;
    #                           broadcasts across the (8, 8) event shape.
    alpha_variational_loc = pyro.param(
        f"{batch_id}_alpha_loc",
        torch.zeros(batch_size, 8, 8, device=device),
    )[batch_idx]
    alpha_variational_scale = pyro.param(
        f"{batch_id}_alpha_scale",
        torch.full((batch_size, 1, 1), 0.1, device=device),
        constraint=dist.constraints.positive,
    )[batch_idx]

    with pyro.plate(f"batch_{batch_id}", batch_size, dim=-1):

        # q(alpha_i) = Normal(mu_i, s_i),  shape (8, 8) via .to_event(2).
        per_loan_logit_offsets = pyro.sample(
            f"alpha_{batch_id}",
            dist.Normal(alpha_variational_loc, alpha_variational_scale).to_event(2)
        )   # (batch_size, 8, 8)

        # Per-loan transition logits theta_i = mask( DEMO_LOGITS + alpha_i ).
        # Deterministic given the sampled alpha_i -- required to propagate the
        # latent delinquency-state sequence through the guide in lockstep with
        # the model so the ELBO can be evaluated at each h_{batch_id}_{t}.
        per_loan_transition_logits = construct_transition_logits(
            per_loan_logit_offsets, device
        )

        # Latent delinquency-state sequence: site names match the model exactly.
        for t in range(1, simulator.max_timesteps + 1):
            next_delinquency_states = pyro.sample(
                f"h_{batch_id}_{t}",
                dist.Categorical(logits=per_loan_transition_logits[
                    torch.arange(batch_size, device=device),
                    simulator.current_hidden_states,
                ])
            )
            simulator.step(next_delinquency_states)


def simulate_portfolio_from_samples(samples, loan, bid, num_samples, num_timesteps, device):
    """
    Reconstruct cash-flow trajectories from posterior (or prior) samples of the
    latent delinquency-state sequence {h_{bid}_t}.

    The simulation is deterministic conditional on the sampled state sequences:
    no additional random variables are drawn here.  Each draw from the posterior
    over {h_{i,t}} induces one payment trajectory; running num_samples draws
    yields a Monte Carlo approximation to the posterior predictive distribution
    over cash flows.

    Parameters
    ----------
    samples       : dict mapping site names to sampled tensors, as returned by
                    a Pyro predictive object.
    loan          : DataLoader batch dict (keys: loan_amnt, installment, int_rate)
                    or a single df row with the same fields as attributes.
    bid           : batch_id string used to look up h_{bid}_{t} in samples.
    num_samples   : number of posterior samples per loan.
    num_timesteps : int or (batch_size,) LongTensor -- observed history lengths T_i.
    device        : torch device string.

    Returns
    -------
    simulator : Portfolio
        Simulator whose payment and balance histories reflect the full set of
        sampled trajectories.  Call .get_histories() for per-period arrays or
        .get_total_pre_chargeoff() / .get_last_payment() for scalar summaries.
    """
    if isinstance(loan, dict):
        # DataLoader batch: repeat each loan's contract terms num_samples times
        # so all posterior draws can be processed in a single vectorised pass.
        loan_amnt    = loan['loan_amnt'].repeat_interleave(num_samples).to(device)
        installments = loan['installment'].repeat_interleave(num_samples).to(device)
        int_rate     = loan['int_rate'].repeat_interleave(num_samples).to(device)
        if torch.is_tensor(num_timesteps):
            num_timesteps_rep = num_timesteps.repeat_interleave(num_samples).to(device)
        else:
            num_timesteps_rep = num_timesteps
    else:
        # Single loan row from df_tmat.
        loan_amnt    = torch.tensor(loan.loan_amnt).repeat(num_samples).to(device)
        installments = torch.tensor(loan.installment).repeat(num_samples).to(device)
        int_rate     = torch.tensor(loan.int_rate).repeat(num_samples).to(device)
        num_timesteps_rep = num_timesteps

    max_t = num_timesteps.max().item() if torch.is_tensor(num_timesteps) else num_timesteps

    # Posterior samples of the latent delinquency-state sequence {h_{bid}_t}.
    # Each element is a (num_samples * batch_size,) tensor of discrete states.
    sampled_state_sequence = [
        samples[f"h_{bid}_{t}"].squeeze(-1).flatten()
        for t in range(1, max_t + 1)
    ]

    simulator = Portfolio(
        loan_amnt=loan_amnt,
        installments=installments,
        int_rate=int_rate,
        num_timesteps=num_timesteps_rep,
        device=device,
    )

    # Propagate each sampled state sequence through the cash-flow simulator.
    # The result is a full posterior predictive distribution over payment
    # trajectories -- deterministic conditional on the sampled {h_{bid}_t}.
    for sampled_states in sampled_state_sequence:
        simulator.step(sampled_states.to(device))

    return simulator
