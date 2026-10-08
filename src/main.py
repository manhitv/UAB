import argparse, os, time, random, json
from collections import Counter
from datetime import datetime
import numpy as np
import torch
from tqdm import tqdm

from model.model_utils import get_agents, engine_vllm_batch
from data.data_utils import load_data
from evaluator import (
    evaluate_math_latex, get_instruction_suffix,
    evaluate_arithmetics, evaluate_mcq,
)
from utils import (
    allocate_extra_budget, allocate_budget_random, allocate_budget_length,
    llm_judge_difficulty, allocate_budget_judge,
)

BASELINES = ('vote', 'random', 'length', 'llm_judge')
LOGPROB_METRICS = {'anll': 0, 'nll': 1, 'token_var': 2, 'min_token_nll': 3}
ROUNDED_ANSWER_DATA = ('gsm8k', 'prog_expr')


def convert_numpy(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Type {type(obj)} not serializable")


def agent_names(args, n, start=0):
    return [f"{args.data}_{args.data_size}__{args.model}__default__Agent{start + j + 1}"
            for j in range(n)]


def user_message(question, suffix):
    return [{"role": "user", "content": question + suffix}]


def get_evaluate(args):
    if args.data in ('gsm8k', 'deepscaler', 'prog_expr'):
        return evaluate_arithmetics
    if args.data == 'formal_logic':
        return evaluate_mcq
    if args.data == 'math500':
        return evaluate_math_latex
    raise NotImplementedError(f"Unknown dataset: {args.data}")


def build_round_data(args, responses, final_resps, debate_resps, is_corr, y,
                     token_stats=None, uncertainty=None, time_taken=None):
    record = {
        'responses':            responses,
        'final_answers':        final_resps,
        'debate_answer':        debate_resps,
        'debate_answer_iscorr': is_corr,
        'answer':               y,
        'token_stats':          token_stats,
        'uncertainty':          uncertainty,
        'time_taken':           time_taken,
        'num_agents':           len(responses),
    }
    if args.data in ROUNDED_ANSWER_DATA:
        record['final_answer_iscorr'] = [p == np.round(y, 1) for p in final_resps]
        record['answer'] = np.round(y, 1)
    else:
        record['final_answer_iscorr'] = [p == y for p in final_resps]
    return record


def log_and_save(args, sample_responses, iscorr_list, fname):
    os.makedirs('out/history', exist_ok=True)
    with open(f'out/history/{fname}.jsonl', 'w') as f:
        for record in sample_responses:
            f.write(json.dumps(record, default=convert_numpy) + '\n')

    arr = np.array(iscorr_list, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    accs = [round(float(a), 4) for a in arr.mean(0)]

    tsv_path = 'out/novote_logs.tsv' if args.no_phase1_vote else f'out/{args.data}_logs.tsv'
    with open(tsv_path, 'a') as f:
        f.write(f"\n{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\t{fname}\t{accs}")
    print(f"out/history/{fname}.jsonl  |  accs={accs}  |  tsv={tsv_path}")


def vote_entropy(answers):
    """Normalized Shannon entropy of the extracted answers, in [0, 1].

    Dividing by `log(len(answers))` makes the value comparable across Phase-1
    sizes `K`. At `K=2` the result is 0 when both answers agree and 1 otherwise.
    """
    counts = np.array(list(Counter(map(str, answers)).values()), dtype=float)
    probs = counts / counts.sum()
    return max(0.0, float(-np.sum(probs * np.log(probs)) / np.log(len(answers))))


def difficulty_scores(args, responses, uncertainties, test_Y, evaluate, K):
    """Reduce the K Phase-1 responses per question to one difficulty score.

    Under `vote_entropy` the score is the normalized entropy of the K extracted
    answers, which needs no log-probabilities. The log-prob signals instead
    average their per-response statistic over the K samples.
    """
    M = len(test_Y)
    if args.uncertainty_mode == 'vote_entropy':
        scores = []
        for i in range(M):
            resp = dict(zip(agent_names(args, K), responses[i * K:(i + 1) * K]))
            scores.append(vote_entropy(evaluate(resp, test_Y[i])[0]))
        return np.array(scores)
    idx = LOGPROB_METRICS[args.uncertainty_mode]
    return np.array([np.mean([uncertainties[i * K + j][idx] for j in range(K)]) for i in range(M)])


def run_baseline(args, agent, test_X, test_Y, evaluate, suffix):
    """Run a fixed-budget baseline: uniform, random, length, or LLM-judge.

    Each allocator decides the per-question counts up front, so all four spend
    exactly `N * M` samples in a single generation pass.
    """
    M, N = len(test_X), args.num_agents
    total_B = N * M

    if args.budget_mode == 'vote':
        alloc = np.full(M, N, dtype=int)
    elif args.budget_mode == 'random':
        alloc = np.array(allocate_budget_random(M, total_B, c_min=1, seed=args.seed), dtype=int)
    elif args.budget_mode == 'length':
        alloc = np.array(allocate_budget_length(test_X, total_B, agent, c_min=1), dtype=int)
    else:
        labels = llm_judge_difficulty(test_X, agent)
        alloc = np.array(allocate_budget_judge(labels, total_B, c_min=1), dtype=int)

    print(f"[{args.budget_mode}] alloc  min={alloc.min()} max={alloc.max()} "
          f"sum={alloc.sum()} (target {total_B})")

    messages = [user_message(test_X[i], suffix) for i in range(M) for _ in range(alloc[i])]
    t0 = time.time()
    all_resp, all_unc, tok_stats = engine_vllm_batch(messages, agent)
    elapsed = time.time() - t0
    print(f"[{args.budget_mode}] inference done in {elapsed:.1f}s")

    sample_responses, iscorr_list, ptr = [], [], 0
    for i, y in tqdm(enumerate(test_Y), total=M, desc="Eval"):
        n = int(alloc[i])
        resp = dict(zip(agent_names(args, n), all_resp[ptr:ptr + n]))
        final_resps, debate_resps, is_corr = evaluate(resp, y)
        sample_responses.append({'0': build_round_data(
            args, resp, final_resps, debate_resps, is_corr, y,
            token_stats=tok_stats[ptr:ptr + n], uncertainty=all_unc[ptr:ptr + n],
            time_taken=elapsed)})
        iscorr_list.append([is_corr])
        ptr += n

    print(f"\n[{args.budget_mode}] Final accuracy: {np.mean([row[0] for row in iscorr_list]):.4f}")
    return sample_responses, iscorr_list


def asc_stop_prob(counts, rng, mc=300):
    """Estimate the probability that the current answer leads at infinite budget.

    Monte-Carlo estimate under a Dirichlet posterior over answer frequencies, as
    in Aggarwal et al. (2023). ASC stops a question once this exceeds the
    threshold.
    """
    alpha = 1.0 + np.asarray(counts, dtype=float)
    top = int(np.argmax(counts))
    draws = rng.dirichlet(alpha, size=mc)
    return float(np.mean(np.argmax(draws, axis=1) == top))


def run_asc(args, agent, test_X, test_Y, evaluate, suffix):
    """Run Adaptive-Consistency: sample one at a time, stop when the vote is stable.

    Being sequential and per-question, its realized budget is an outcome rather
    than an input: `--num_agents` acts only as a per-question cap, and
    `--asc_thresh` has to be tuned so the average budget matches the target N.
    """
    M, cap = len(test_X), args.num_agents
    rng = np.random.default_rng(args.seed)

    active = np.arange(M)
    resp_dicts = [{} for _ in range(M)]
    tok_lists = [[] for _ in range(M)]
    unc_lists = [[] for _ in range(M)]
    n_seen = np.zeros(M, dtype=int)
    gen_time = 0.0

    print(f"\n[ASC] cap={cap}  thresh={args.asc_thresh}  M={M}")
    for t in range(cap):
        if len(active) == 0:
            break
        print(f"\n=== ASC iter {t + 1}/{cap} | active={len(active)} ===")
        t0 = time.time()
        resp, unc, tok = engine_vllm_batch([user_message(test_X[i], suffix) for i in active], agent)
        gen_time += time.time() - t0

        still_active = []
        for k, i in enumerate(active):
            resp_dicts[i][agent_names(args, 1, start=t)[0]] = resp[k]
            unc_lists[i].append(unc[k])
            tok_lists[i].append(tok[k])
            n_seen[i] += 1

            if n_seen[i] < 2:
                still_active.append(i)
                continue

            final_resps, _, _ = evaluate(resp_dicts[i], test_Y[i])
            parsed = [x for x in final_resps if x != ""]
            if not parsed:
                still_active.append(i)
                continue
            counts = list(Counter(parsed).values())
            if len(counts) > 1 and asc_stop_prob(counts, rng) < args.asc_thresh:
                still_active.append(i)
        active = np.array(still_active, dtype=int)

    sample_responses, iscorr_list = [], []
    for i, y in tqdm(enumerate(test_Y), total=M, desc="Eval ASC"):
        final_resps, debate_resps, is_corr = evaluate(resp_dicts[i], y)
        sample_responses.append({'0': build_round_data(
            args, resp_dicts[i], final_resps, debate_resps, is_corr, y,
            token_stats=tok_lists[i], uncertainty=unc_lists[i])})
        iscorr_list.append([is_corr])

    print(f"\n[ASC] Final accuracy: {np.mean([row[0] for row in iscorr_list]):.4f}  |  "
          f"avg realized N: {n_seen.mean():.2f}  |  gen_time: {gen_time:.1f}s")
    return sample_responses, iscorr_list


def run_uab(args, agent, test_X, test_Y, evaluate, suffix):
    """Run UAB: probe every question with K samples, then allocate the rest.

    Phase 1 draws K samples per question and scores difficulty from their
    agreement. Phase 2 spends the remaining `(N - K) * M` samples by
    marginal-greedy allocation. Phase-1 samples also count toward the final
    majority vote, so the total cost is exactly `N * M`.
    """
    M, N, K = len(test_X), args.num_agents, args.phase1_samples
    if args.uncertainty_mode == 'vote_entropy' and K < 2:
        raise ValueError("vote_entropy requires --phase1_samples >= 2")
    if K > N:
        raise ValueError(f"--phase1_samples ({K}) cannot exceed --num_agents ({N})")

    print(f"\n=== Phase 1: {K} response(s)/question (signal={args.uncertainty_mode}) ===")
    t0 = time.time()
    p1_resp, p1_unc, p1_tok = engine_vllm_batch(
        [user_message(x, suffix) for x in test_X for _ in range(K)], agent)
    p1_time = time.time() - t0

    scores = difficulty_scores(args, p1_resp, p1_unc, test_Y, evaluate, K)
    print(f"[{args.uncertainty_mode}]  min={scores.min():.4f}  max={scores.max():.4f}  mean={scores.mean():.4f}")

    extra_alloc = np.array(allocate_extra_budget(
        scores, (N - K) * M, base_n=K, tau=args.tau), dtype=int)
    print(f"Extra alloc dist: {np.bincount(extra_alloc).tolist()}  total_extra={extra_alloc.sum()}")

    t0 = time.time()
    ex_resp, ex_unc, ex_tok = engine_vllm_batch(
        [user_message(test_X[i], suffix) for i in range(M) for _ in range(extra_alloc[i])], agent) \
        if extra_alloc.sum() > 0 else ([], [], [])
    p2_time = time.time() - t0
    print(f"Phase-1 time {p1_time:.1f}s  |  Phase-2 time {p2_time:.1f}s")

    sample_responses, iscorr_list, ptr = [], [], 0
    for i, y in tqdm(enumerate(test_Y), total=M, desc="Eval"):
        n_ex = int(extra_alloc[i])
        p1 = dict(zip(agent_names(args, K), p1_resp[i * K:(i + 1) * K]))
        extra = dict(zip(agent_names(args, n_ex, start=K), ex_resp[ptr:ptr + n_ex]))

        p1_final, p1_debate, p1_corr = evaluate(p1, y)
        voters = extra if (args.no_phase1_vote and n_ex > 0) else {**p1, **extra}
        final_resps, debate_resps, is_corr = evaluate(voters, y)

        record = build_round_data(
            args, voters, final_resps, debate_resps, is_corr, y,
            token_stats=p1_tok[i * K:(i + 1) * K] + ex_tok[ptr:ptr + n_ex],
            uncertainty=p1_unc[i * K:(i + 1) * K] + ex_unc[ptr:ptr + n_ex],
            time_taken=p1_time + p2_time)
        record['diff_score'] = float(scores[i])
        sample_responses.append({'0': record})
        iscorr_list.append([p1_corr, is_corr])
        ptr += n_ex

    print(f"\nPhase-1 acc: {np.mean([r[0] for r in iscorr_list]):.4f}  |  "
          f"Final acc: {np.mean([r[1] for r in iscorr_list]):.4f}")
    return sample_responses, iscorr_list


def get_args():
    p = argparse.ArgumentParser(description="UAB: Uncertainty-Aware Budget allocation")

    p.add_argument('--seed',      type=int, default=42)
    p.add_argument('--data',      type=str, default='')
    p.add_argument('--data_size', type=int, default=0)
    p.add_argument('--split',     type=str, default='test')
    p.add_argument('--debug',     action='store_true')

    p.add_argument('--model',      type=str, default="qwen2.5-1.5b")
    p.add_argument('--num_agents', type=int, default=4)
    p.add_argument('--cot',        action='store_true')

    p.add_argument('--budget_mode', type=str, default='uav_base',
                   choices=list(BASELINES) + ['asc', 'uav_base'])
    p.add_argument('--tau',        type=float, default=1.0)
    p.add_argument('--asc_thresh', type=float, default=0.95)
    p.add_argument('--phase1_samples', type=int, default=2)
    p.add_argument('--uncertainty_mode', type=str, default='vote_entropy',
                   choices=['vote_entropy'] + list(LOGPROB_METRICS))
    p.add_argument('--no_phase1_vote', action='store_true')

    return p.parse_args()


def make_fname(args):
    fname = (f"{args.data}__{args.data_size}__{args.model}"
             f"__N={args.num_agents}__M={args.budget_mode}__S={args.seed}")
    if args.budget_mode == 'uav_base':
        fname += f"__T={args.tau}__UM={args.uncertainty_mode}__P1={args.phase1_samples}"
    if args.budget_mode == 'asc':
        fname += f"__ASC={args.asc_thresh}"
    if args.no_phase1_vote:
        fname += "__NOVOTE"
    if args.debug:
        fname = "DEBUG_" + fname
    return fname


def main(args):
    test_X, test_Y = load_data(args, split=args.split)[:2]
    if args.debug:
        test_X, test_Y = test_X[:8], test_Y[:8]

    suffix = get_instruction_suffix(args)
    evaluate = get_evaluate(args)
    fname = make_fname(args)

    print(f"\nExperiment : {fname}")
    print(f"Samples    : {len(test_X)}  |  N={args.num_agents}  |  mode={args.budget_mode}")

    agent = get_agents(args)
    if args.budget_mode in BASELINES:
        run = run_baseline
    elif args.budget_mode == 'asc':
        run = run_asc
    else:
        run = run_uab
    sample_responses, iscorr_list = run(args, agent, test_X, test_Y, evaluate, suffix)

    log_and_save(args, sample_responses, iscorr_list, fname)
    print("DONE:", fname)


if __name__ == "__main__":
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    start = datetime.now()
    main(args)
    print(f"======================\nTotal Time: {datetime.now() - start}\n======================")
