import numpy as np
from vllm import SamplingParams


_JUDGE_SYSTEM = """\
Classify whether the following question is easy or hard for a language model to answer correctly.
- Easy: the answer is likely correct with a single attempt (simple fact, direct reasoning).
- Hard: likely requires multiple attempts or careful reasoning to get right.

Respond with exactly one digit — 1 (Easy) or 2 (Hard) — and nothing else."""


_JUDGE_USER_TMPL = "Question: {question}\nDifficulty (1=Easy / 2=Hard):"


def allocate_extra_budget(diff_scores, total_extra_budget, base_n=1, tau=1.0):
    p = np.exp(-np.asarray(diff_scores, dtype=float) / tau)
    extra = np.zeros(len(p), dtype=int)
    for _ in range(total_extra_budget):
        marginal = p * (1 - p) ** (base_n + extra)
        extra[np.argmax(marginal)] += 1
    return extra.tolist()


def allocate_budget_random(
    M: int,
    total_budget: int,
    c_min: int = 1,
    seed: int = 42,
) -> list[int]:
    rng = np.random.default_rng(seed)

    B_eff = total_budget - M * c_min

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
    M = len(questions)
    assert total_budget >= M * c_min, "Budget too small for c_min constraint."

    tokenizer = agent.llm.get_tokenizer()
    token_counts = np.array([
        len(tokenizer.encode(q, add_special_tokens=False))
        for q in questions
    ], dtype=float)

    if token_counts.max() == token_counts.min():
        base = total_budget // M
        budgets = np.full(M, base, dtype=int)
        budgets[:total_budget - base * M] += 1
        return budgets.tolist()

    B_eff = total_budget - M * c_min

    w = token_counts / token_counts.sum()

    c_star = c_min + B_eff * w
    floors = np.floor(c_star).astype(int)
    deficit = total_budget - floors.sum()
    floors[:deficit] += 1

    assert floors.sum() == total_budget
    assert (floors >= c_min).all()
    assert token_counts.min() > 0, "Empty question detected."

    return floors.tolist()


def llm_judge_difficulty(
    questions: list[str],
    agent,
) -> list[int]:
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
        temperature=0,
        max_tokens=4,
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
            label = 2
        labels.append(label)

    return labels


def allocate_budget_judge(
    labels: list[int],
    total_budget: int,
    c_min: int = 1,
) -> list[int]:
    labels = np.array(labels)
    M = len(labels)
    assert total_budget >= M * c_min, "Budget too small for c_min constraint."

    easy_mask = labels == 1
    hard_mask = labels == 2
    n_easy = int(easy_mask.sum())
    n_hard = int(hard_mask.sum())

    budgets = np.full(M, c_min, dtype=int)

    if n_hard == 0:
        surplus = total_budget - M * c_min
        budgets += surplus // M
        budgets[:surplus % M] += 1
    else:
        B_hard = total_budget - n_easy * c_min
        c_hard_base = B_hard // n_hard
        c_hard_rem  = B_hard %  n_hard

        budgets[hard_mask] = c_hard_base
        hard_indices = np.where(hard_mask)[0]
        budgets[hard_indices[:c_hard_rem]] += 1

    assert budgets.sum() == total_budget
    assert (budgets >= c_min).all()
    return budgets.tolist()
