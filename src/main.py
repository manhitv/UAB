# main.py — UAB: Uncertainty-Aware Budget allocation for test-time scaling
# =============================================================================
# Single entry point for every experiment in the paper.
#
#   Baselines      : --budget_mode {vote,random,length,llm_judge}
#   UAB (Ours)     : --budget_mode uav_base
#   Budget scaling : --num_agents N
#   Threshold exit : --hard_mode / --easy_mode  (with --theta_hard / --theta_easy)
#   Uncertainty    : --diff_metric {anll,nll,token_var,min_token_nll}
#   Verbalized conf: --uncertainty_mode vcs
#   Phase-1 vote   : --no_phase1_vote
#   Black-box API  : --backend cohere  (--budget_mode {vote,llm_judge,uav})
#   Analysis tables: --analyze
#
# Each run writes out/history/<fname>.jsonl (responses, token stats, timing)
# and appends accuracy to out/<data>_logs.tsv.
# =============================================================================

import argparse, os, re, time, random, json
import numpy as np
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import torch

from model.model_utils import get_agents, engine_vllm_batch
from data.data_utils import load_data
from evaluator import (
    evaluate_math_competition, evaluate_math_latex, get_instruction_suffix,
    evaluate_arithmetics, evaluate_mcq,
    base_evaluate_arithmetics, base_evaluate_mcq, evaluate_hle
)
from utils import (
    allocate_budget, allocate_extra_budget,
    allocate_budget_random, allocate_budget_length,
    llm_judge_difficulty, allocate_budget_judge,
)

BASELINE_METHODS = frozenset({'vote', 'random', 'length', 'llm_judge'})
ADAPTIVE_MODES   = frozenset({'uav_base', 'uav'})

# Uncertainty tuple from the engine: (anll, nll, token_var, min_token_nll)
DIFF_METRIC_IDX = {'anll': 0, 'nll': 1, 'token_var': 2, 'min_token_nll': 3}


# ── Helpers ───────────────────────────────────────────────────────────────────

def convert_numpy(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Type {type(obj)} not serializable")


def _extract_diff_score(unc, metric='anll'):
    """Pick the difficulty signal from an engine uncertainty tuple."""
    idx = DIFF_METRIC_IDX.get(metric, 0)
    if isinstance(unc, (list, tuple)) and len(unc) > idx:
        return float(unc[idx])
    return float(unc)


def get_sample_agent_names(n_agents, personas, args):
    pkeys = list(personas.keys()) if personas else []
    return [
        f"{args.data}_{args.data_size}__{args.model}"
        f"__{pkeys[j % len(pkeys)] if pkeys else 'default'}__Agent{j+1}"
        for j in range(n_agents)
    ]


def _build_round_data(args, agent_responses, final_resps, debate_resps,
                      is_corr, y, token_stats=None, uncertainty=None,
                      time_taken=None, n_agents=None):
    base = {
        'responses':            agent_responses,
        'final_answers':        final_resps,
        'debate_answer':        debate_resps,
        'debate_answer_iscorr': is_corr,
        'answer':               y,
        'token_stats':          token_stats,
        'uncertainty':          uncertainty,
        'time_taken':           time_taken,
        'num_agents':           n_agents,
    }
    if args.data in ['arithmetics', 'gsm8k']:
        base['final_answer_iscorr'] = [yp == np.round(y, 1) for yp in final_resps]
        base['answer'] = np.round(y, 1)
    else:
        base['final_answer_iscorr'] = [yp == y for yp in final_resps]
    return base


def _get_evaluate(args):
    if args.data in ['arithmetics', 'gsm8k', 'deepscaler']:
        return base_evaluate_arithmetics if args.bae else evaluate_arithmetics
    if args.data in ['hellaswag', 'pro_medicine', 'formal_logic',
                     'csqa', 'hh_rlhf', 'mmlu_pro', 'gpqa']:
        return base_evaluate_mcq if args.bae else evaluate_mcq
    if args.data in ['aime24', 'aime25', 'amc23']:
        return evaluate_math_competition
    if args.data in ['math500', 'minerva', 'math_algebra']:
        return evaluate_math_latex
    if args.data == 'hle':
        return evaluate_hle
    raise NotImplementedError(f"Unknown dataset: {args.data}")


def _save_history(sample_responses, fname):
    os.makedirs('out/history', exist_ok=True)
    with open(f'out/history/{fname}.jsonl', 'w') as f:
        for record in sample_responses:
            f.write(json.dumps(record, default=convert_numpy) + '\n')


def _log_and_save(args, sample_responses, iscorr_list, fname):
    _save_history(sample_responses, fname)

    arr = np.array(iscorr_list, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    accs = [round(float(a), 4) for a in arr.mean(0)]

    if getattr(args, 'uncertainty_mode', 'anll') == 'vcs':
        tsv_path = 'out/vcs_logs.tsv'
    elif getattr(args, 'no_phase1_vote', False):
        tsv_path = 'out/novote_logs.tsv'
    else:
        tsv_path = f'out/{args.data}_logs.tsv'

    os.makedirs('out', exist_ok=True)
    with open(tsv_path, 'a') as f:
        ts = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        f.write(f"\n{ts}\t{fname}\t{accs}")

    print(f"  → out/history/{fname}.jsonl  |  accs={accs}  |  tsv={tsv_path}")
    return accs


# ── Verbalized Confidence Score (VCS) ─────────────────────────────────────────
# The confidence-elicitation instruction is appended directly to the question,
# so the model emits its answer and a 1-10 confidence rating in a single Phase-1
# generation.  The parsed integer is normalised to a per-sample success
# probability p_i = score / 10.

VCS_PROMPT = (
    "\n\nAfter giving your answer, rate how confident you are that it is "
    "correct on a scale from 1 (not confident at all) to 10 (completely "
    "certain). End your response with:\nConfidence: <integer>."
)


def parse_vcs(text):
    """Parse the trailing 'Confidence: <int>' rating; return p = score/10 in [0,1].

    Falls back to a neutral 0.5 when no rating can be parsed.
    """
    matches = re.findall(r'[Cc]onfidence:\s*([0-9]+)', str(text))
    if not matches:
        return 0.5
    score = min(max(int(matches[-1]), 1), 10)
    return score / 10.0


# ── Threshold-exit helpers ────────────────────────────────────────────────────

def classify_easy_samples(anll_scores, theta_easy, tau=1.0, score_fn='exp'):
    """Easy: per-attempt success probability p_i > theta_easy."""
    s = np.asarray(anll_scores, dtype=float)
    p_i = np.exp(-s / tau) if score_fn == 'exp' else 1.0 / (1.0 + np.exp(s / tau))
    return p_i > theta_easy, p_i


def classify_hard_samples(anll_scores, N_full, theta_hard, tau=1.0, score_fn='exp'):
    """Too hard: 1 - (1-p_i)^N_full < theta_hard."""
    s = np.asarray(anll_scores, dtype=float)
    p_i = np.exp(-s / tau) if score_fn == 'exp' else 1.0 / (1.0 + np.exp(s / tau))
    prob_success = 1.0 - np.power(np.clip(1.0 - p_i, 0.0, 1.0), N_full)
    return prob_success < theta_hard, p_i, prob_success


def _log_threshold(tsv_path, header, ts_row):
    os.makedirs('out', exist_ok=True)
    write_header = not os.path.exists(tsv_path)
    with open(tsv_path, 'a') as f:
        if write_header:
            f.write(header + '\n')
        f.write(ts_row + '\n')


def _log_hard_threshold(args, fname, hard_mask, budget_total, budget_used,
                        iscorr_list, accs):
    M = len(hard_mask)
    n_hard    = int(hard_mask.sum())
    hard_frac = round(n_hard / M, 4) if M > 0 else 0.0
    final_iscorr = np.array([row[-1] for row in iscorr_list], dtype=float)
    acc_all     = round(float(final_iscorr.mean()), 4)
    acc_nonhard = round(float(final_iscorr[~hard_mask].mean()), 4) if (~hard_mask).any() else float('nan')
    acc_hard    = round(float(final_iscorr[hard_mask].mean()),  4) if hard_mask.any()    else float('nan')

    header = ("timestamp\tfname\tdata\tmodel\tN\ttheta_hard\thard_mode"
              "\tn_hard\thard_frac\tbudget_total\tbudget_used\tbudget_saved"
              "\tacc_all\tacc_nonhard\tacc_hard\taccs_checkpoints")
    ts = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    row = (f"{ts}\t{fname}\t{args.data}\t{args.model}"
           f"\t{args.num_agents}\t{args.theta_hard}\t{args.hard_mode}"
           f"\t{n_hard}\t{hard_frac}\t{budget_total}\t{budget_used}\t{budget_total - budget_used}"
           f"\t{acc_all}\t{acc_nonhard}\t{acc_hard}\t{accs}")
    _log_threshold('out/hard_threshold.tsv', header, row)
    print(f"  [hard_threshold] n_hard={n_hard}/{M} ({hard_frac:.1%})  "
          f"budget_saved={budget_total - budget_used}  acc_all={acc_all}  "
          f"acc_nonhard={acc_nonhard}  acc_hard={acc_hard}")


def _log_easy_threshold(args, fname, easy_mask, budget_total, budget_used,
                        iscorr_list, accs):
    M = len(easy_mask)
    n_easy    = int(easy_mask.sum())
    easy_frac = round(n_easy / M, 4) if M > 0 else 0.0
    final_iscorr = np.array([row[-1] for row in iscorr_list], dtype=float)
    acc_all     = round(float(final_iscorr.mean()), 4)
    acc_noneasy = round(float(final_iscorr[~easy_mask].mean()), 4) if (~easy_mask).any() else float('nan')
    acc_easy    = round(float(final_iscorr[easy_mask].mean()),  4) if easy_mask.any()    else float('nan')

    header = ("timestamp\tfname\tdata\tmodel\tN\ttheta_easy\teasy_mode"
              "\tn_easy\teasy_frac\tbudget_total\tbudget_used\tbudget_saved"
              "\tacc_all\tacc_noneasy\tacc_easy\taccs_checkpoints")
    ts = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    row = (f"{ts}\t{fname}\t{args.data}\t{args.model}"
           f"\t{args.num_agents}\t{args.theta_easy}\t{args.easy_mode}"
           f"\t{n_easy}\t{easy_frac}\t{budget_total}\t{budget_used}\t{budget_total - budget_used}"
           f"\t{acc_all}\t{acc_noneasy}\t{acc_easy}\t{accs}")
    _log_threshold('out/easy_threshold.tsv', header, row)
    print(f"  [easy_threshold] n_easy={n_easy}/{M} ({easy_frac:.1%})  "
          f"budget_saved={budget_total - budget_used}  acc_all={acc_all}  "
          f"acc_noneasy={acc_noneasy}  acc_easy={acc_easy}")


def analyze_hard_threshold(theta_hard_vals=(0.3, 0.5, 0.7, 0.9), tau=1.0, save=True):
    """Print theoretical tables for the threshold-exit criterion 1-(1-p)^B."""
    p_i_vals = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70]
    B_vals   = (1, 2, 4, 8, 16, 32)
    lines, sep = [], "=" * 72

    lines += ["\n" + sep, "Hard Threshold Analysis:  P(>=1 success) = 1 - (1 - p_i)^B", sep]

    lines.append("\nTable 1: P(at least 1 correct) = 1-(1-p_i)^B")
    hdr = f"  {'p_i':<7}" + "".join(f"{'B='+str(b):<9}" for b in B_vals)
    lines += [hdr, "  " + "-" * (len(hdr) - 2)]
    for p in p_i_vals:
        lines.append(f"  {p:<7.3f}" + "".join(f"{1-(1-p)**b:<9.4f}" for b in B_vals))

    lines.append("\nTable 2: Minimum p_i so that 1-(1-p_i)^B > theta_hard")
    hdr2 = f"  {'theta':<9}" + "".join(f"{'B='+str(b):<9}" for b in B_vals)
    lines += [hdr2, "  " + "-" * (len(hdr2) - 2)]
    for theta in theta_hard_vals:
        row = f"  {theta:<9.2f}"
        for b in B_vals:
            row += f"{1.0 - (1.0 - theta) ** (1.0 / b):<9.4f}"
        lines.append(row)

    lines.append(f"\nTable 3: Equivalent ANLL threshold  [tau={tau}]  anll = -tau*ln(p_min)")
    hdr3 = f"  {'theta':<9}" + "".join(f"{'B='+str(b):<9}" for b in B_vals)
    lines += [hdr3, "  " + "-" * (len(hdr3) - 2)]
    for theta in theta_hard_vals:
        row = f"  {theta:<9.2f}"
        for b in B_vals:
            p_min = 1.0 - (1.0 - theta) ** (1.0 / b)
            anll_thr = -tau * np.log(p_min) if p_min > 1e-12 else np.inf
            row += f"{anll_thr:<9.3f}"
        lines.append(row)

    lines.append("\n" + sep + "\n")
    text = "\n".join(lines)
    print(text)
    if save:
        os.makedirs('out', exist_ok=True)
        with open('out/hard_threshold_analysis.txt', 'w') as fh:
            fh.write(text)
        print("  Saved to out/hard_threshold_analysis.txt")
    return text


# ── Cohere black-box backend ──────────────────────────────────────────────────

COHERE_MAX_RETRIES = 5
COHERE_AUTH_KWS    = ("bearer", "401", "unauthorized", "api key", "invalid token", "forbidden")

_JUDGE_PROMPT_BB = """\
Classify the difficulty of the following question for an AI model to answer correctly.

Question: {question}

Reply with ONLY: 1 (Easy) or 2 (Hard)."""


def _init_cohere(api_key, model):
    import cohere
    key = api_key or os.environ.get('COHERE_API_KEY', '')
    if not key:
        raise ValueError(
            "Cohere API key not found. Use --cohere_api_key KEY "
            "or export COHERE_API_KEY=KEY")
    print(f"[Cohere] model={model}")
    return cohere.ClientV2(api_key=key), model


def _cohere_call(messages, client, model):
    last_exc = None
    for attempt in range(COHERE_MAX_RETRIES):
        try:
            resp = client.chat(messages=messages, model=model)
            return resp.message.content[0].text
        except Exception as exc:
            last_exc = exc
            if any(kw in str(exc).lower() for kw in COHERE_AUTH_KWS):
                print(f"[Cohere] Auth error (not retrying): {exc}")
                break
            wait = 2 ** attempt
            print(f"[Cohere] Attempt {attempt+1}/{COHERE_MAX_RETRIES}, retry in {wait}s: {exc}")
            time.sleep(wait)
    print(f"[Cohere] Failed after {COHERE_MAX_RETRIES} attempts: {last_exc}")
    return ""


def cohere_batch(messages_list, client, model):
    return [_cohere_call(m, client, model)
            for m in tqdm(messages_list, desc="Cohere", leave=False)]


def cohere_sample_per_question(grouped, client, model):
    """For each (message, n) fire n concurrent calls; flat list out."""
    results = []
    for msg, n in tqdm(grouped, desc="Questions"):
        if n <= 1:
            results.append(_cohere_call(msg, client, model))
        else:
            with ThreadPoolExecutor(max_workers=n) as ex:
                futures = [ex.submit(_cohere_call, msg, client, model) for _ in range(n)]
                results.extend(f.result() for f in futures)
    return results


def get_llm_judge_labels(questions, client, model):
    msgs = [[{"role": "user", "content": _JUDGE_PROMPT_BB.format(question=q)}]
            for q in questions]
    return [1 if (r.strip() and r.strip()[0] == '1') else 2
            for r in cohere_batch(msgs, client, model)]


def uav_allocate_extra(vcs, extra_budget, tau=1.0):
    """UAB greedy marginal allocation from verbalized confidence (p_i = vcs^(1/tau))."""
    if extra_budget <= 0:
        return [0] * len(vcs)
    p = np.clip(np.array(vcs, dtype=float) ** (1.0 / tau), 1e-9, 1.0 - 1e-9)
    c = np.zeros(len(vcs), dtype=int)
    for _ in range(extra_budget):
        gain = p * (1.0 - p) ** c
        c[int(np.argmax(gain))] += 1
    return c.tolist()


def judge_allocate(labels, total_budget, c_min=1):
    """Binary easy/hard allocation: hard questions share the remaining budget."""
    M = len(labels)
    alloc = np.full(M, c_min, dtype=int)
    n_hard = sum(l == 2 for l in labels)
    if n_hard == 0:
        for k in range(total_budget - M * c_min):
            alloc[k % M] += 1
        return alloc.tolist()
    B_hard_pool = total_budget - (M - n_hard) * c_min
    per_hard = max(c_min, B_hard_pool // n_hard)
    for i, lbl in enumerate(labels):
        if lbl == 2:
            alloc[i] = per_hard
    diff = int(total_budget - alloc.sum())
    hard_idx = [i for i, l in enumerate(labels) if l == 2]
    for k in range(abs(diff)):
        alloc[hard_idx[k % n_hard]] += 1 if diff > 0 else -1
    return alloc.tolist()


# ── Local backend: single-round baselines ────────────────────────────────────

def _run_baseline_single(args, agent, personas, test_X, test_Y, evaluate, SUFFIX):
    """Generate the allocated number of responses per sample in one flat batch."""
    M, N = len(test_X), args.num_agents
    total_B = N * M

    if args.budget_mode == 'vote':
        alloc = np.full(M, N, dtype=int)
    elif args.budget_mode == 'random':
        alloc = np.array(allocate_budget_random(M, total_B, c_min=1, seed=args.seed), dtype=int)
    elif args.budget_mode == 'length':
        alloc = np.array(allocate_budget_length(test_X, total_B, agent, c_min=1), dtype=int)
    elif args.budget_mode == 'llm_judge':
        labels = llm_judge_difficulty(test_X, agent)
        alloc  = np.array(allocate_budget_judge(labels, total_B, c_min=1), dtype=int)
    else:
        raise ValueError(f"Unknown baseline: {args.budget_mode}")

    print(f"[{args.budget_mode}] alloc  min={alloc.min()} max={alloc.max()} "
          f"sum={alloc.sum()} (target {total_B})")

    persona_values = list(personas.values()) if personas else []
    messages = []
    for i in range(M):
        for j in range(alloc[i]):
            if args.multi_persona and persona_values:
                messages.append([
                    {"role": "system", "content": persona_values[j % len(persona_values)]},
                    {"role": "user",   "content": test_X[i] + SUFFIX},
                ])
            else:
                messages.append([{"role": "user", "content": test_X[i] + SUFFIX}])

    t0 = time.time()
    all_resp, all_unc, tok_stats = engine_vllm_batch(messages, agent)
    elapsed = time.time() - t0
    print(f"[{args.budget_mode}] inference done in {elapsed:.1f}s")

    sample_responses = [{} for _ in range(M)]
    iscorr_list      = []
    ptr = 0
    for i, (x, y) in tqdm(enumerate(zip(test_X, test_Y)), total=M, desc="Eval"):
        n     = int(alloc[i])
        resps = all_resp[ptr : ptr + n]
        unc_i = all_unc[ptr : ptr + n] if all_unc else None
        ts_i  = tok_stats[ptr : ptr + n] if tok_stats else None
        ptr  += n

        names = get_sample_agent_names(n, personas, args)
        agent_resp_dict = dict(zip(names, resps))
        final_resps, debate_resps, is_corr = evaluate(agent_resp_dict, y)

        sample_responses[i]['0'] = _build_round_data(
            args, agent_resp_dict, final_resps, debate_resps, is_corr, y,
            token_stats=ts_i, uncertainty=unc_i, time_taken=elapsed, n_agents=n)
        iscorr_list.append([is_corr])

    print(f"\n[{args.budget_mode}] Final accuracy: "
          f"{np.mean([row[0] for row in iscorr_list]):.4f}")
    return sample_responses, iscorr_list


# ── Local backend: UAB two-phase pipeline ─────────────────────────────────────

def _run_uab(args, agent, personas, test_X, test_Y, evaluate, SUFFIX):
    """
    UAB two-phase pipeline.
      Phase 1 — one response/sample → difficulty signal (ANLL or VCS).
      Phase 2 — allocate (N-1)*M extra responses by marginal-greedy, then vote.

    Supports optional hard/easy threshold exits; returns the masks plus
    realised budget so the caller can log threshold statistics.
    """
    M, N = len(test_X), args.num_agents
    persona_values = list(personas.values()) if personas else []
    hard_mode = args.hard_mode
    easy_mode = args.easy_mode
    unc_mode  = args.uncertainty_mode
    no_p1_vote = args.no_phase1_vote

    # Phase 1.  In VCS mode the confidence instruction is appended to the
    # question so the model answers and self-rates in a single call.
    print(f"\n=== Phase 1: 1 response/sample (uncertainty_mode={unc_mode}) ===")
    vcs_suffix = VCS_PROMPT if unc_mode == 'vcs' else ''
    r0_messages = []
    for x in test_X:
        if args.multi_persona and persona_values:
            r0_messages.append([
                {"role": "system", "content": persona_values[0]},
                {"role": "user",   "content": x + SUFFIX + vcs_suffix},
            ])
        else:
            r0_messages.append([{"role": "user", "content": x + SUFFIX + vcs_suffix}])

    t_p1 = time.time()
    all_r0_resp, all_r0_unc, r0_tok = engine_vllm_batch(r0_messages, agent)
    p1_time = time.time() - t_p1

    # Difficulty signal
    if unc_mode == 'vcs':
        vcs = np.array([parse_vcs(r) for r in all_r0_resp])
        diff_scores = 1.0 - vcs            # higher difficulty = lower confidence
        print(f"VCS   min={vcs.min():.2f}  max={vcs.max():.2f}  mean={vcs.mean():.3f}"
              f"  fallback={int((vcs == 0.5).sum())}/{len(vcs)}")
    else:
        diff_scores = np.array([_extract_diff_score(u, args.diff_metric) for u in all_r0_unc])
        print(f"[{args.diff_metric}]  min={diff_scores.min():.4f}  "
              f"max={diff_scores.max():.4f}  mean={diff_scores.mean():.4f}")

    # Threshold classification
    N_max = args.n_max if args.n_max > 0 else N
    hard_mask = np.zeros(M, dtype=bool)
    easy_mask = np.zeros(M, dtype=bool)
    prob_succ = np.zeros(M)

    if hard_mode != 'none':
        hard_mask, p_i_all, prob_succ = classify_hard_samples(
            diff_scores, N_full=N_max, theta_hard=args.theta_hard,
            tau=args.tau, score_fn=args.score_fn)
        print(f"[Hard threshold] theta_hard={args.theta_hard}  N_full={N_max}  "
              f"n_hard={int(hard_mask.sum())}/{M}")
    if easy_mode != 'none':
        easy_mask, _ = classify_easy_samples(
            diff_scores, theta_easy=args.theta_easy,
            tau=args.tau, score_fn=args.score_fn)
        print(f"[Easy threshold] theta_easy={args.theta_easy}  "
              f"n_easy={int(easy_mask.sum())}/{M}")

    # Budget allocation
    skip_mask    = hard_mask | easy_mask
    non_skip_idx = np.where(~skip_mask)[0]
    n_non_skip   = len(non_skip_idx)
    active_mode  = hard_mode if hard_mode != 'none' else easy_mode
    budget_total = N * M

    if active_mode == 'none':
        extra_budget = (N - 1) * M
        if args.budget_mode in ADAPTIVE_MODES:
            extra_alloc = allocate_extra_budget(
                diff_scores=diff_scores, total_extra_budget=extra_budget,
                strategy=args.budget_mode, **vars(args))
        else:
            raw_alloc = allocate_budget(
                scores=diff_scores, total_budget=N * M, c_min=1,
                strategy=args.alloc_mode, **vars(args))
            extra_alloc = np.maximum(np.array(raw_alloc, dtype=int) - 1, 0)
        extra_alloc = np.array(extra_alloc, dtype=int)
    else:
        # Skipped questions get no Phase-2 budget.
        # redistribute: full (N-1)*M goes to non-skip → total calls = N*M.
        # skip:         only (N-1)*n_non_skip is spent → total calls < N*M.
        extra_budget = (N - 1) * M if active_mode == 'redistribute' else (N - 1) * n_non_skip
        extra_alloc  = np.zeros(M, dtype=int)
        if n_non_skip > 0:
            extra_alloc[non_skip_idx] = np.array(allocate_extra_budget(
                diff_scores=diff_scores[non_skip_idx],
                total_extra_budget=extra_budget, strategy='uav_base',
                tau=args.tau, score_fn=args.score_fn, N_max=N_max,
                theta_easy=args.theta_easy), dtype=int)

    budget_used = M + int(extra_alloc.sum())
    print(f"Extra alloc dist: {np.bincount(extra_alloc).tolist()}  total_extra={extra_alloc.sum()}")
    print(f"Budget: total={budget_total}  used={budget_used}  saved={budget_total - budget_used}")

    # Evaluate Phase 1
    print(f"\n=== Evaluating Phase 1 (no_phase1_vote={no_p1_vote}) ===")
    sample_responses        = [{} for _ in range(M)]
    iscorr_list             = [[] for _ in range(M)]
    history_agent_responses = [{} for _ in range(M)]

    for i, (x, y) in tqdm(enumerate(zip(test_X, test_Y)), total=M, desc="Eval P1"):
        names      = get_sample_agent_names(1, personas, args)
        agent_resp = dict(zip(names, [all_r0_resp[i]]))
        history_agent_responses[i] = {} if no_p1_vote else agent_resp
        final_resps, debate_resps, is_corr = evaluate(agent_resp, y)
        sample_responses[i]['0'] = _build_round_data(
            args, agent_resp, final_resps, debate_resps, is_corr, y,
            token_stats=[r0_tok[i]]     if r0_tok     else None,
            uncertainty=[all_r0_unc[i]] if all_r0_unc else None,
            time_taken=p1_time, n_agents=1)
        sample_responses[i]['0']['is_hard']      = bool(hard_mask[i])
        sample_responses[i]['0']['is_easy']      = bool(easy_mask[i])
        sample_responses[i]['0']['prob_success'] = float(prob_succ[i])
        iscorr_list[i].append(is_corr)

    # Phase 2
    p2_time = 0.0
    if extra_alloc.sum() > 0:
        print(f"\n=== Phase 2: {extra_alloc.sum()} extra responses "
              f"for {int((extra_alloc > 0).sum())} samples ===")
        extra_msgs = []
        for i in range(M):
            for j in range(extra_alloc[i]):
                if args.multi_persona and persona_values:
                    extra_msgs.append([
                        {"role": "system", "content": persona_values[j % len(persona_values)]},
                        {"role": "user",   "content": test_X[i] + SUFFIX},
                    ])
                else:
                    extra_msgs.append([{"role": "user", "content": test_X[i] + SUFFIX}])

        t_p2 = time.time()
        ex_resp, ex_unc, _ = engine_vllm_batch(extra_msgs, agent)
        p2_time = time.time() - t_p2

        ptr = 0
        for i in range(M):
            n_ex = int(extra_alloc[i])
            if n_ex == 0:
                if no_p1_vote:
                    # No Phase-2 sample and Phase-1 excluded: fall back to Phase-1.
                    history_agent_responses[i] = dict(zip(
                        get_sample_agent_names(1, personas, args), [all_r0_resp[i]]))
                continue
            if no_p1_vote:
                history_agent_responses[i] = {}
                base_name = get_sample_agent_names(1, personas, args)[0]
            else:
                base_name = list(history_agent_responses[i].keys())[0]
            for j, resp in enumerate(ex_resp[ptr : ptr + n_ex], 1):
                history_agent_responses[i][f"{base_name}_extra{j}"] = resp
            ptr += n_ex

            all_d = history_agent_responses[i]
            final_resps, debate_resps, is_corr = evaluate(all_d, test_Y[i])
            rec = sample_responses[i]['0']
            rec['responses']            = all_d
            rec['final_answers']        = final_resps
            rec['debate_answer']        = debate_resps
            rec['debate_answer_iscorr'] = is_corr
            rec['num_agents']           = len(all_d)
            rec['time_taken']           = p1_time + p2_time
            if args.data in ['arithmetics', 'gsm8k']:
                rec['final_answer_iscorr'] = [p == np.round(test_Y[i], 1) for p in final_resps]
            else:
                rec['final_answer_iscorr'] = [p == test_Y[i] for p in final_resps]
            if ex_unc:
                rec['uncertainty'].extend(ex_unc[ptr - n_ex : ptr])
            iscorr_list[i].append(is_corr)

    r0_acc    = np.mean([iscorr_list[i][0]  for i in range(M)])
    final_acc = np.mean([iscorr_list[i][-1] for i in range(M)])
    print(f"\nPhase-1 acc: {r0_acc:.4f}  |  Final acc: {final_acc:.4f}  "
          f"|  time: P1={p1_time:.1f}s P2={p2_time:.1f}s")

    iscorr_flat = [[iscorr_list[i][0], iscorr_list[i][-1]] for i in range(M)]
    return sample_responses, iscorr_flat, hard_mask, easy_mask, budget_total, budget_used


# ── Cohere backend: vote / llm_judge / uab ────────────────────────────────────

def _run_vote_cohere(args, client, model, test_X, test_Y, evaluate, SUFFIX):
    M, N = len(test_X), args.num_agents
    grouped = [([{"role": "user", "content": x + SUFFIX}], N) for x in test_X]
    print(f"\n[vote] {N} x {M} = {N*M} calls")
    t0 = time.time()
    all_resp = cohere_sample_per_question(grouped, client, model)
    elapsed = time.time() - t0
    print(f"[vote] done in {elapsed:.1f}s")

    sample_responses, iscorr_list = [{} for _ in range(M)], []
    for i, (_, y) in tqdm(enumerate(zip(test_X, test_Y)), total=M, desc="Eval"):
        agent_d = dict(zip(get_sample_agent_names(N, None, args), all_resp[i*N:(i+1)*N]))
        final_resps, debate_resps, is_corr = evaluate(agent_d, y)
        sample_responses[i]['0'] = _build_round_data(
            args, agent_d, final_resps, debate_resps, is_corr, y,
            time_taken=elapsed, n_agents=N)
        iscorr_list.append([is_corr])
    print(f"\n[vote] Accuracy: {np.mean([r[0] for r in iscorr_list]):.4f}")
    return sample_responses, iscorr_list


def _run_llm_judge_cohere(args, client, model, test_X, test_Y, evaluate, SUFFIX):
    M, N = len(test_X), args.num_agents
    print(f"\n[llm_judge] classifying {M} samples...")
    labels = get_llm_judge_labels(test_X, client, model)
    print(f"[llm_judge] easy={sum(l == 1 for l in labels)}  hard={sum(l == 2 for l in labels)}")
    alloc = np.array(judge_allocate(labels, total_budget=N*M, c_min=1), dtype=int)

    grouped = [([{"role": "user", "content": test_X[i] + SUFFIX}], int(alloc[i]))
               for i in range(M)]
    t0 = time.time()
    all_resp = cohere_sample_per_question(grouped, client, model)
    elapsed = time.time() - t0
    print(f"[llm_judge] generated {alloc.sum()} responses in {elapsed:.1f}s")

    sample_responses, iscorr_list = [{} for _ in range(M)], []
    ptr = 0
    for i, (_, y) in tqdm(enumerate(zip(test_X, test_Y)), total=M, desc="Eval"):
        n = int(alloc[i])
        agent_d = dict(zip(get_sample_agent_names(n, None, args), all_resp[ptr:ptr+n]))
        ptr += n
        final_resps, debate_resps, is_corr = evaluate(agent_d, y)
        sample_responses[i]['0'] = _build_round_data(
            args, agent_d, final_resps, debate_resps, is_corr, y,
            time_taken=elapsed, n_agents=n)
        iscorr_list.append([is_corr])
    print(f"\n[llm_judge] Accuracy: {np.mean([r[0] for r in iscorr_list]):.4f}")
    return sample_responses, iscorr_list


def _run_uab_cohere(args, client, model, test_X, test_Y, evaluate, SUFFIX):
    """Black-box UAB: VCS difficulty signal + marginal-greedy allocation."""
    M, N = len(test_X), args.num_agents
    extra_budget = (N - 1) * M

    print(f"\n[UAB] Phase 1: {M} probe responses (answer + confidence in one call)...")
    t0 = time.time()
    probe_resps = cohere_batch(
        [[{"role": "user", "content": x + SUFFIX + VCS_PROMPT}] for x in test_X],
        client, model)
    vcs = [parse_vcs(r) for r in probe_resps]
    p1_time = time.time() - t0
    print(f"[UAB] VCS mean={np.mean(vcs):.3f}  Phase-1 done in {p1_time:.1f}s")

    extra_alloc = np.array(uav_allocate_extra(vcs, extra_budget, tau=args.tau), dtype=int)
    print(f"[UAB] extra alloc  min={extra_alloc.min()} max={extra_alloc.max()} "
          f"sum={extra_alloc.sum()} (target {extra_budget})")

    sample_responses  = [{} for _ in range(M)]
    iscorr_list       = [[] for _ in range(M)]
    history_responses = [{} for _ in range(M)]
    for i, (_, y) in tqdm(enumerate(zip(test_X, test_Y)), total=M, desc="Eval P1"):
        agent_d = dict(zip(get_sample_agent_names(1, None, args), [probe_resps[i]]))
        history_responses[i] = agent_d
        final_resps, debate_resps, is_corr = evaluate(agent_d, y)
        sample_responses[i]['0'] = _build_round_data(
            args, agent_d, final_resps, debate_resps, is_corr, y,
            uncertainty=[vcs[i]], time_taken=p1_time, n_agents=1)
        iscorr_list[i].append(is_corr)

    p2_time = 0.0
    if extra_alloc.sum() > 0:
        print(f"\n[UAB] Phase 2: {extra_alloc.sum()} extra calls...")
        grouped = [([{"role": "user", "content": test_X[i] + SUFFIX}], int(extra_alloc[i]))
                   for i in range(M) if extra_alloc[i] > 0]
        t2 = time.time()
        extra_resps = cohere_sample_per_question(grouped, client, model)
        p2_time = time.time() - t2

        ptr = 0
        for i in range(M):
            n_ex = int(extra_alloc[i])
            if n_ex == 0:
                continue
            base_name = list(history_responses[i].keys())[0]
            for j, resp in enumerate(extra_resps[ptr:ptr+n_ex], 1):
                history_responses[i][f"{base_name}_extra{j}"] = resp
            ptr += n_ex
            all_d = history_responses[i]
            final_resps, debate_resps, is_corr = evaluate(all_d, test_Y[i])
            rec = sample_responses[i]['0']
            rec['responses']            = all_d
            rec['final_answers']        = final_resps
            rec['debate_answer']        = debate_resps
            rec['debate_answer_iscorr'] = is_corr
            rec['num_agents']           = len(all_d)
            rec['time_taken']           = p1_time + p2_time
            if args.data in ['arithmetics', 'gsm8k']:
                rec['final_answer_iscorr'] = [p == np.round(test_Y[i], 1) for p in final_resps]
            else:
                rec['final_answer_iscorr'] = [p == test_Y[i] for p in final_resps]
            iscorr_list[i].append(is_corr)

    print(f"\n[UAB] Phase-1 acc={np.mean([iscorr_list[i][0] for i in range(M)]):.4f}  "
          f"Final acc={np.mean([iscorr_list[i][-1] for i in range(M)]):.4f}")
    iscorr_flat = [[iscorr_list[i][0], iscorr_list[i][-1]] for i in range(M)]
    return sample_responses, iscorr_flat


# ── Args ──────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="UAB — Uncertainty-Aware Budget allocation")

    p.add_argument('--seed',      type=int, default=42)
    p.add_argument('--out_dir',   type=str, default="out/")

    p.add_argument('--data',      type=str, default='')
    p.add_argument('--sub_data',  type=str, default='')
    p.add_argument('--data_size', type=int, default=0)
    p.add_argument('--split',     type=str, default='test')
    p.add_argument('--debug',     action='store_true',
                   help="Run on a small slice and prefix the output filename with DEBUG_")

    p.add_argument('--model',          type=str, default="qwen2.5-1.5b")
    p.add_argument('--num_agents',     type=int, default=4,
                   help="Per-question budget N (total responses per question)")
    p.add_argument('--multi_persona',  action='store_true')
    p.add_argument('--bae',            action='store_true')
    p.add_argument('--cot',            action='store_true')

    p.add_argument('--backend', type=str, default='local', choices=['local', 'cohere'],
                   help="local = vLLM/HF open-weight; cohere = black-box Cohere API")
    p.add_argument('--cohere_api_key', type=str, default='',
                   help="Cohere API key (falls back to $COHERE_API_KEY)")

    p.add_argument('--budget_mode', type=str, default='uav_base',
                   choices=['vote', 'random', 'length', 'llm_judge', 'uav', 'uav_base'],
                   help="vote/random/length/llm_judge = baselines; "
                        "uav_base = UAB (Ours); uav = UAB on the Cohere backend")
    p.add_argument('--tau',        type=float, default=0.2,
                   help="Softmax temperature for budget allocation")
    p.add_argument('--p',          type=float, default=2.0)
    p.add_argument('--q',          type=float, default=0.5)
    p.add_argument('--alloc_mode', type=str,   default='softmax')
    p.add_argument('--score_fn',   type=str,   default='exp', choices=['exp', 'sigmoid'])
    p.add_argument('--n_max',      type=int,   default=0,
                   help="Full-budget cap for hard-question classification (0 = use N)")

    p.add_argument('--theta_easy', type=float, default=0.5)
    p.add_argument('--theta_hard', type=float, default=0.5)
    p.add_argument('--hard_mode',  type=str, default='none',
                   choices=['none', 'redistribute', 'skip'],
                   help="Threshold exit for too-hard questions")
    p.add_argument('--easy_mode',  type=str, default='none',
                   choices=['none', 'redistribute', 'skip'],
                   help="Threshold exit for easy questions")

    p.add_argument('--diff_metric', type=str, default='anll',
                   choices=['anll', 'nll', 'token_var', 'min_token_nll'],
                   help="Log-prob-derived difficulty signal for UAB allocation")
    p.add_argument('--uncertainty_mode', type=str, default='anll',
                   choices=['anll', 'vcs'],
                   help="anll = log-prob difficulty; vcs = verbalized confidence")
    p.add_argument('--no_phase1_vote', action='store_true',
                   help="Exclude the Phase-1 generation from the majority vote")

    p.add_argument('--analyze', action='store_true',
                   help="Print threshold-exit analysis tables and exit")

    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def _make_fname(args):
    fname = (f"{args.data}__{args.data_size}__{args.model}"
             f"__N={args.num_agents}__M={args.budget_mode}"
             f"__T={args.tau}__S={args.seed}")
    if args.budget_mode in ADAPTIVE_MODES or args.budget_mode == 'uav':
        fname += f"__EASY={args.theta_easy}__HARD={args.theta_hard}__N_MAX={args.n_max}__FN={args.score_fn}"
    if args.hard_mode != 'none':         fname += f"__HM={args.hard_mode}"
    if args.easy_mode != 'none':         fname += f"__EM={args.easy_mode}"
    if args.diff_metric != 'anll':       fname += f"__DM={args.diff_metric}"
    if args.uncertainty_mode != 'anll':  fname += f"__UM={args.uncertainty_mode}"
    if args.no_phase1_vote:              fname += "__NOVOTE"
    if args.backend == 'cohere':         fname += "__BLACKBOX"
    if args.bae:                         fname += "_BAE"
    if args.multi_persona:               fname += "_HETERO"
    if args.debug:                       fname = "DEBUG_" + fname
    return fname


def main(args):
    if args.analyze:
        analyze_hard_threshold(tau=args.tau, save=True)
        return

    if args.data in ['deepscaler', 'math500', 'math_algebra', 'prog_expr']:
        test_X, test_Y, _ = load_data(args, split='test')
    else:
        test_X, test_Y = load_data(args, split='test')
    if args.debug:
        test_X, test_Y = test_X[:8], test_Y[:8]

    SUFFIX   = get_instruction_suffix(args)
    evaluate = _get_evaluate(args)
    M        = len(test_X)
    fname    = _make_fname(args)
    args.fname = fname

    print(f"\nExperiment : {fname}")
    print(f"Samples    : {M}  |  N={args.num_agents}  |  backend={args.backend}  "
          f"|  mode={args.budget_mode}")

    hard_mask    = np.zeros(M, dtype=bool)
    easy_mask    = np.zeros(M, dtype=bool)
    budget_total = args.num_agents * M
    budget_used  = budget_total

    if args.backend == 'cohere':
        client, model = _init_cohere(args.cohere_api_key, args.model)
        if args.budget_mode in ('vote', 'random', 'length'):
            sample_responses, iscorr_list = _run_vote_cohere(
                args, client, model, test_X, test_Y, evaluate, SUFFIX)
        elif args.budget_mode == 'llm_judge':
            sample_responses, iscorr_list = _run_llm_judge_cohere(
                args, client, model, test_X, test_Y, evaluate, SUFFIX)
        else:
            sample_responses, iscorr_list = _run_uab_cohere(
                args, client, model, test_X, test_Y, evaluate, SUFFIX)
    else:
        agent, personas = get_agents(args)
        if args.budget_mode in BASELINE_METHODS:
            sample_responses, iscorr_list = _run_baseline_single(
                args, agent, personas, test_X, test_Y, evaluate, SUFFIX)
        else:
            sample_responses, iscorr_list, hard_mask, easy_mask, budget_total, budget_used = \
                _run_uab(args, agent, personas, test_X, test_Y, evaluate, SUFFIX)

    accs = _log_and_save(args, sample_responses, iscorr_list, fname)

    if args.hard_mode != 'none':
        _log_hard_threshold(args, fname, hard_mask, budget_total, budget_used,
                            iscorr_list, accs)
    if args.easy_mode != 'none':
        _log_easy_threshold(args, fname, easy_mask, budget_total, budget_used,
                            iscorr_list, accs)
    print("DONE:", fname)


if __name__ == "__main__":
    args = get_args()

    if args.analyze:
        analyze_hard_threshold(tau=args.tau, save=True)
    else:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        start = datetime.now()
        args.timestamp = start.strftime("%d/%m/%Y %H:%M:%S")
        main(args)
        print(f"======================\nTotal Time: {datetime.now() - start}\n======================")
