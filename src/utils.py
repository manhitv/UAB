"""utils.py — budget-allocation algorithms for UAB.

Difficulty-weighted allocation strategies (marginal-greedy / closed-form /
softmax) plus the baseline allocators (random, length, LLM-judge).
"""

import numpy as np
from vllm import SamplingParams


_JUDGE_SYSTEM = """\
Classify whether the following question is easy or hard for a language model to answer correctly.
- Easy: the answer is likely correct with a single attempt (simple fact, direct reasoning).
- Hard: likely requires multiple attempts or careful reasoning to get right.

Respond with exactly one digit — 1 (Easy) or 2 (Hard) — and nothing else."""


_JUDGE_USER_TMPL = "Question: {question}\nDifficulty (1=Easy / 2=Hard):"


def compute_weights(s: np.ndarray, strategy: str = "borda", **kwargs) -> np.ndarray:
    M = len(s)

    if strategy == "softmax":
        w = np.exp(s / kwargs.get("tau", 1.0))

    elif strategy == "borda":
        # rank 1 = hardest, borda score = M
        ranks = np.argsort(np.argsort(-s)) + 1   # 1..M, 1=hardest
        p = kwargs.get("p", 2.0)
        w = (M - ranks + 1) ** p                  # hardest → borda = M

    elif strategy == "percentile":
        q = kwargs.get("q", 50)
        mask = s >= np.percentile(s, q)
        w = np.where(mask, np.exp(s / kwargs.get("tau", 1.0)), 0.0)

    elif strategy == "power":
        w = np.clip(s, 0, None) ** kwargs.get("alpha", 2.0)
        
    elif strategy == "sigmoid":
        # adaptive threshold via percentile
        q = kwargs.get("q", 0.7) * 100   # default: top 30%  hard (50 -> 80)
        d = np.percentile(s, q)
        temp = kwargs.get("temp", 0.1) # sharpness of gating (0.05 -> 0.1)
        tau = kwargs.get("tau", 1.0)    # softmax temperature

        # sigmoid gating
        g = 1 / (1 + np.exp(-(s - d) / temp))   # ∈ (0,1)

        # base score weight
        w_base = np.exp(s / tau)

        # combine
        w = g * w_base

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    total = w.sum()
    return w / total if total > 0 else np.ones_like(s) / M


def score_to_p(s, tau=1.0):
    # p ~ exp(-ANLL/tau): high ANLL (hard) -> low p, low ANLL (easy) -> high p
    return 1 / (1 + np.exp(s / tau))


def score_to_p_exp(s, tau=1.0):
    # alternative: p ~ exp(-ANLL/tau) directly
    return np.exp(-s / tau)


def closed_form_allocate(p, B, c_min=1, max_iter=50):
    """
    Solve:
        sum_i N_i = B
        N_i >= c_min
    via Lagrangian root finding on lambda
    """

    M = len(p)
    log1m = np.log1p(-p)

    # shift baseline (enforce c_min)
    B_eff = B - M * c_min

    # initial lambda bounds
    lo, hi = 1e-12, 1.0

    def compute_N(lam):
        N = np.log(lam / p) / log1m
        N = np.maximum(0.0, N)
        return N

    # binary search on lambda
    for _ in range(max_iter):
        lam = np.sqrt(lo * hi)
        N = compute_N(lam)
        total = np.sum(N)

        if total > B_eff:
            lo = lam
        else:
            hi = lam

    N = compute_N(hi)

    # add c_min
    N = N + c_min

    # integer projection
    floors = np.floor(N).astype(int)
    rem = N - floors

    deficit = B - floors.sum()
    if deficit > 0:
        idx = np.argsort(-rem)[:deficit]
        floors[idx] += 1

    return floors


def allocate_budget(
    scores: np.ndarray,
    total_budget: int,
    c_min: int = 1,
    strategy: str = "rank",
    **strategy_kwargs,
) -> list[int]:
    """
    Allocate integer LLM call counts C_i with:
      - sum(C_i) == total_budget
      - C_i >= c_min  (for non-skipped questions)

    Strategies:
      Closed-form:
        - closed_form       : exact optimal under Bernoulli model
      Marginal greedy:
        - marginal_prob     : greedy argmax p*(1-p)^c, score_to_p sigmoid
        - marginal_prob_exp : same but score_to_p_exp
      Early-exit + greedy:
        - marginal_early_exit : early exit for easy/hard + greedy on rest
            params:
              tau         (float, default 1.0)   — score→p sharpness
              N_max       (int,   default None)  — cap per question; None = total_budget
              theta_easy  (float, default 0.9)   — p_i > θ → assign c_min, done
              theta_hard  (float, default 0.5)   — P(correct|N_max) < θ → skip
              score_fn    (str,   default "exp") — "exp" | "sigmoid"
      Heuristic:
        - softmax    : softmax(scores/tau)
        - rank       : rank-based, param: alpha (default 2.0)
        - percentile : top-q% gets extra, params: q (default 50), tau (default 1.0)
        - power      : score^alpha, param: alpha (default 2.0)

    Returns:
      list[int] of length M with sum == total_budget, each >= c_min
      (skipped questions in marginal_early_exit get 0)
    """
    M = len(scores)
    s = np.array(scores, dtype=float)

    # ------------------------------------------------------------------
    # 0. CLOSED-FORM
    # ------------------------------------------------------------------
    if strategy == "closed_form":
        assert total_budget >= M * c_min
        tau      = strategy_kwargs.get("tau",      1.0)
        score_fn = strategy_kwargs.get("score_fn", "exp")
        p = score_to_p_exp(s, tau=tau) if score_fn == "exp" else score_to_p(s, tau=tau)
        return closed_form_allocate(p, total_budget, c_min=c_min).tolist()

    # ------------------------------------------------------------------
    # 1. MARGINAL GREEDY (original variants)
    # ------------------------------------------------------------------
    if strategy in ("marginal_prob", "marginal_prob_exp", "uav_base"):
        assert total_budget >= M * c_min
        tau = strategy_kwargs.get("tau", 1.0)
        score_fn   = strategy_kwargs.get("score_fn", "exp")
        
        p = score_to_p_exp(s, tau=tau) if score_fn == "exp" else score_to_p(s, tau=tau)

        c = np.full(M, c_min, dtype=int)
        for _ in range(total_budget - M * c_min):
            marginal = p * (1 - p) ** c
            c[np.argmax(marginal)] += 1
        return c.tolist()

    # ------------------------------------------------------------------
    # 2. EARLY-EXIT + MARGINAL GREEDY  ← new strategy
    # ------------------------------------------------------------------
    if strategy in ["marginal_early_exit", "uav_early_exit"]:
        tau        = strategy_kwargs.get("tau",        1.0)
        N_max      = strategy_kwargs.get("N_max",      None)
        theta_easy = strategy_kwargs.get("theta_easy", 0.5) # 0.7, 0.8
        theta_hard = strategy_kwargs.get("theta_hard", 0.5) # 0.3, 0.4
        score_fn   = strategy_kwargs.get("score_fn",   "exp")

        if N_max is None:
            N_max = total_budget - 1  # fallback: uncapped (minus 1 for estimation)

        p = score_to_p_exp(s, tau=tau) if score_fn == "exp" else score_to_p(s, tau=tau)

        c       = np.zeros(M, dtype=int)
        is_easy = np.zeros(M, dtype=bool)
        is_hard = np.zeros(M, dtype=bool)

        # ── Pass 1: classify each question ──────────────────────────
        for i in range(M):
            if p[i] > theta_easy:
                is_easy[i] = True
                c[i] = c_min                    # 1 sample enough
            elif (1 - (1 - p[i]) ** N_max) < theta_hard:
                is_hard[i] = True
                c[i] = 0                        # skip; budget freed
            # else: active — handled in Pass 2

        # ── Budget accounting ────────────────────────────────────────
        # 1 estimation sample already spent on EVERY question (AvgNLL)
        B_used   = 0                            # estimation cost
        B_used  += int(c[is_easy].sum())        # easy questions
        # hard questions contribute 0 generation samples
        B_left   = total_budget - B_used

        if B_left < 0:
            raise ValueError(
                f"Budget exhausted after estimation+easy: need {B_used}, "
                f"got {total_budget}. Increase total_budget or theta_easy."
            )

        # ── Pass 2: greedy marginal on active questions ──────────────
        active_idx = np.where(~is_easy & ~is_hard)[0]

        if len(active_idx) > 0 and B_left > 0:
            c_act = np.full(len(active_idx), c_min, dtype=int)
            p_act = p[active_idx]
            B_left -= len(active_idx) * c_min  # assign c_min baseline

            if B_left < 0:
                # Not enough budget even for c_min on all active — 
                # fund as many as possible in order of highest p
                priority = np.argsort(-p_act)
                c_act[:] = 0
                remaining = total_budget - M - int(c[is_easy].sum())
                for idx in priority:
                    if remaining <= 0:
                        break
                    c_act[idx] = c_min
                    remaining -= c_min
                B_left = 0

            for _ in range(max(0, B_left)):
                marginal = np.where(
                    c_act < N_max,
                    p_act * (1 - p_act) ** c_act,
                    -np.inf,
                )
                best = int(np.argmax(marginal))
                if marginal[best] <= 0:
                    break
                c_act[best] += 1

            c[active_idx] = c_act

        return c.tolist()

    # ------------------------------------------------------------------
    # 3. HEURISTIC (weight-based)
    # ------------------------------------------------------------------
    assert total_budget >= M * c_min
    B_eff = total_budget - M * c_min
    w = compute_weights(s, strategy=strategy, **strategy_kwargs)

    c_star   = c_min + B_eff * w
    floors   = np.floor(c_star).astype(int)
    deficit  = total_budget - floors.sum()
    indices  = np.argsort(-(c_star - floors))[:deficit]
    floors[indices] += 1

    assert floors.sum() == total_budget
    assert (floors >= c_min).all()
    return floors.tolist()


def allocate_extra_budget(
    diff_scores,
    total_extra_budget: int,
    strategy: str = "marginal_prob",
    **strategy_kwargs,
) -> list[int]:
    """
    Allocate EXTRA responses beyond Round 0.

    Round 0 always generates exactly 1 response per sample (used for AvgNLL
    estimation and kept in the final vote).  This function distributes the
    remaining (N-1)*M budget with c_min=0, so easy samples can legitimately
    receive 0 extra responses and rely solely on their R0 answer.

    Args:
        diff_scores        : AvgNLL scores from Round 0, one per sample.
        total_extra_budget : (num_agents - 1) * num_samples
        strategy           : same choices as allocate_budget
        **strategy_kwargs  : forwarded to allocate_budget (tau, theta_easy, …)

    Returns:
        list[int] of length M, each entry >= 0, summing to total_extra_budget.
    """
    return allocate_budget(
        scores=np.array(diff_scores, dtype=float),
        total_budget=total_extra_budget,
        c_min=0,
        strategy=strategy,
        **strategy_kwargs,
    )


def allocate_budget_random(
    M: int,
    total_budget: int,
    c_min: int = 1,
    seed: int = 42,
) -> list[int]:
    """
    Random allocation: directly sample integer budgets,
    không qua compute_weights để tránh softmax collapse.
    
    Strategy: sample Dirichlet → continuous weights → Hamilton rounding
    """
    rng = np.random.default_rng(seed)

    B_eff = total_budget - M * c_min

    # Dirichlet(1,...,1) = uniform over simplex → true random allocation
    w = rng.dirichlet(np.ones(M))

    c_star = c_min + B_eff * w
    floors = np.floor(c_star).astype(int)
    deficit = total_budget - floors.sum()
    remainders = c_star - floors
    floors[np.argsort(-remainders)[:deficit]] += 1

    assert floors.sum() == total_budget
    return floors.tolist()


def allocate_budget_length(
    questions: list[str],
    total_budget: int,
    agent,
    c_min: int = 1,
) -> list[int]:
    """
    Length-based allocation: longer question → more budget.
    Uses linear proportional allocation (budget ∝ token_count) 
    with Hamilton rounding, bypasses compute_weights/softmax.

    Args:
        questions    : list of raw question strings
        total_budget : total number of LLM calls to allocate
        agent        : object exposing agent.llm (vLLM LLM instance)
        c_min        : minimum budget per question

    Returns:
        budgets : list[int], one integer budget per question
    """
    M = len(questions)
    assert total_budget >= M * c_min, "Budget too small for c_min constraint."

    tokenizer = agent.llm.get_tokenizer()
    token_counts = np.array([
        len(tokenizer.encode(q, add_special_tokens=False))
        for q in questions
    ], dtype=float)

    # Edge case: all questions same length → uniform
    if token_counts.max() == token_counts.min():
        base = total_budget // M
        budgets = np.full(M, base, dtype=int)
        budgets[:total_budget - base * M] += 1
        return budgets.tolist()

    B_eff = total_budget - M * c_min

    # Linear proportional weights (no softmax)
    w = token_counts / token_counts.sum()

    # Continuous allocation + Hamilton rounding
    c_star = c_min + B_eff * w
    floors = np.floor(c_star).astype(int)
    # remainders = c_star - floors
    deficit = total_budget - floors.sum()
    floors[:deficit] += 1 # do not sort by remainders

    assert floors.sum() == total_budget
    assert (floors >= c_min).all()
    assert token_counts.min() > 0, "Empty question detected."

    return floors.tolist()


def llm_judge_difficulty(
    questions: list[str],
    agent,
) -> list[int]:
    """
    Binary easy/hard classification via a single greedy LLM call per question.
    Consistent with Difficulty-Aware Self-Consistency (Yue et al., 2024).

    Returns:
        labels : list[int], 1 = Easy, 2 = Hard.
                 Falls back to Hard (2) on parse errors (conservative).
    """
    tokenizer = agent.llm.get_tokenizer()

    prompts = []
    for q in questions:
        messages = [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user",   "content": _JUDGE_USER_TMPL.format(question=q)},
        ]
        prompts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )

    sampling_params = SamplingParams(
        temperature=0,   # greedy — deterministic, reproducible
        max_tokens=4,    # single digit token
        logprobs=None,
    )

    outputs = agent.llm.generate(prompts, sampling_params, use_tqdm=True)

    labels = []
    for out in outputs:
        raw = out.outputs[0].text.strip()
        try:
            label = int(raw[0])
            if label not in (1, 2):
                raise ValueError
        except (ValueError, IndexError):
            label = 2  # conservative fallback: treat ambiguous as hard
        labels.append(label)

    return labels


def allocate_budget_judge(
    labels: list[int],
    total_budget: int,
    c_min: int = 1,
) -> list[int]:
    """
    Binary budget allocation matching Difficulty-Aware Self-Consistency:
      easy (1) → c_min  (lock at minimum)
      hard (2) → remaining budget split equally, Hamilton-rounded

    No free hyperparameters. Exactly satisfies sum(budgets) == total_budget.
    """
    labels = np.array(labels)
    M = len(labels)
    assert total_budget >= M * c_min, "Budget too small for c_min constraint."

    easy_mask = labels == 1
    hard_mask = labels == 2
    n_easy = int(easy_mask.sum())
    n_hard = int(hard_mask.sum())

    budgets = np.full(M, c_min, dtype=int)

    if n_hard == 0:
        # All easy: distribute surplus uniformly
        surplus = total_budget - M * c_min
        budgets += surplus // M
        budgets[:surplus % M] += 1
    else:
        # Hard questions share everything beyond what easy questions need
        B_hard = total_budget - n_easy * c_min   # guaranteed >= n_hard * c_min
        c_hard_base = B_hard // n_hard
        c_hard_rem  = B_hard %  n_hard

        budgets[hard_mask] = c_hard_base
        hard_indices = np.where(hard_mask)[0]
        budgets[hard_indices[:c_hard_rem]] += 1

    assert budgets.sum() == total_budget
    assert (budgets >= c_min).all()
    return budgets.tolist()
