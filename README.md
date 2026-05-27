<div align="center">
<img src="assets/uab_logo.svg" height=140 alt="UAB">
  <h1><b> UAB: Uncertainty-Aware Budget Allocation </b></h1>
  <p><i>Spend test-time compute where it matters — adaptive sampling for LLM reasoning.</i></p>
</div>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2605.26849-b31b1b)](https://arxiv.org/pdf/2605.26849)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![vLLM](https://img.shields.io/badge/Powered_by-vLLM-orange)](https://github.com/vllm-project/vllm)

</div>

<div align="center">

🚀 [**Getting Started**](#install) **|**
🔧 [**Usage**](#usage) **|**
🧪 [**Reproducing Results**](#reproduce) **|**
🎯 [**Benchmarks**](#bench) **|**
📂 [**Project Structure**](#structure)

</div>

**UAB** redistributes a *fixed* sampling budget across questions by difficulty — at
**zero extra cost**. Equal-sample baselines waste compute on easy inputs and starve
hard ones; UAB casts the allocation as a **concave integer program** solved in two
phases:

- **Phase 1 — Probe.** One generation per question; its **average negative
  log-likelihood (ANLL)** from the output log-probs is the difficulty signal. This
  sample also counts toward the final majority vote.
- **Phase 2 — Allocate.** A **marginal-greedy** algorithm distributes the remaining
  `(N-1)·M` budget, exactly maximizing a concave coverage surrogate — more samples to
  uncertain questions, fewer to confident ones.

No auxiliary model, no extra LLM call. Works on open-weight models (ANLL from
log-probs) and black-box APIs (verbalized confidence).

📜 **Paper:** [*Uncertainty-Aware Budget Allocation for Adaptive Test-Time Reasoning*](https://arxiv.org/pdf/2605.26849).

> If you find this repository helpful for your work, please consider citing as follows:
>
> ```LaTeX
> @misc{nguyen2026uncertaintyawarebudgetallocationadaptive,
>     title={Uncertainty-Aware Budget Allocation for Adaptive Test-Time Reasoning}, 
>      author={Manh Nguyen and Sunil Gupta and Hung Le},
>      year={2026},
>      eprint={2605.26849},
>      archivePrefix={arXiv},
>      primaryClass={cs.CL},
>      url={https://arxiv.org/abs/2605.26849}, 
> }
> ```
---

## <a name="install"></a> 🚀 Installation

#### ⏬ Environment Setup

```bash
git clone https://github.com/manhitv/UAB.git UAB
cd UAB

conda create -n uab python=3.11 -y
conda activate uab
pip install -r requirements.txt
```

> 📌 **Black-box experiments (Cohere)?** Export your API key:
> ```bash
> export COHERE_API_KEY="your_cohere_api_key_here"
> ```

---

## <a name="usage"></a> 🔧 Usage

`src/main.py` is the **single entry point for every experiment**. The method is selected
entirely by CLI flags.

**Run UAB on MATH-500 with a per-question budget of `N=4`:**

```bash
python src/main.py --model qwen2.5-1.5b --data math500 --data_size 500 \
  --budget_mode uav_base --num_agents 4 --tau 0.2 --seed 42
```

**Run a baseline (uniform self-consistency) under the same budget:**

```bash
python src/main.py --model qwen2.5-1.5b --data math500 --data_size 500 \
  --budget_mode vote --num_agents 4 --seed 42
```

#### Key flags

| Flag | Description |
|------|-------------|
| `--budget_mode`  | `vote`, `random`, `length`, `llm_judge` (baselines) or `uav_base` (**UAB**) |
| `--num_agents N` | Per-question budget `N` (total responses per question) |
| `--tau`          | Softmax temperature for allocation (default `0.2`) |
| `--data` / `--data_size` | Benchmark name and number of test questions |
| `--model`        | Model key (see [Benchmarks](#bench)) |
| `--seed`         | Random seed (paper averages over `42 44 46`) |
| `--backend cohere` | Black-box Cohere API instead of local inference |

#### ⚡ Quick validation

Run a short end-to-end check of the full pipeline:

```bash
bash scripts/validate.sh
```

#### 📒 Outputs

After each run:
- **Accuracy** is appended to `out/<dataset>_logs.tsv` (one row per run).
- **Full history** (responses, ANLL/uncertainty, per-response token counts, wall-clock
  time, per-question allocation) is serialized to `out/history/<experiment_name>.jsonl`.
- Threshold-exit runs additionally append to `out/hard_threshold.tsv` /
  `out/easy_threshold.tsv`; VCS and no-vote ablations log to `out/vcs_logs.tsv` /
  `out/novote_logs.tsv`.
- `--debug` runs on a small slice and prefixes the filename with `DEBUG_`.

---

## <a name="reproduce"></a> 🧪 Reproducing Paper Results

Every experiment in the paper is driven by `src/main.py`. The `scripts/` directory
bundles the full sweeps; representative commands are shown below. All main results are
averaged over **seeds `42 44 46`**.

| Script | Paper result |
|--------|--------------|
| `scripts/run_main_table.sh`  | Table 1 — main accuracy (5 models × 5 benchmarks, `N=4`) |
| `scripts/run_scaling.sh`     | Budget scaling `N∈{2,4,8}` and cost-at-fixed-accuracy `N∈{1..16}` |
| `scripts/run_blackbox.sh`    | Black-box & big-model table (GPT-OSS-20B, Gemma3-27B, Cohere) |
| `scripts/run_ablations.sh`   | Temperature, uncertainty metric, threshold exits, Phase-1 vote, VCS, wall-clock |
| `scripts/validate.sh`        | Quick end-to-end smoke test |

---

## <a name="bench"></a> 🎯 Benchmarks

#### ☝️ Tested Models

| Model key | Hugging Face / API |
|-----------|--------------------|
| `qwen2.5-1.5b`     | `Qwen/Qwen2.5-1.5B-Instruct` |
| `qwen2.5-7b`       | `Qwen/Qwen2.5-7B-Instruct`   |
| `llama3.2-3b`      | `meta-llama/Llama-3.2-3B-Instruct` |
| `gptoss`           | `openai/gpt-oss-20b`         |
| `gemma3-27b`       | `google/gemma-3-27b-it`      |
| `command-a-03-2025`| Cohere API (black-box, `--backend cohere`) |

#### ✌️ Supported Benchmarks

| `--data` | `--data_size` | Task |
|----------|---------------|------|
| `deepscaler`   | 300 | Competition math |
| `math500`      | 500 | Competition math |
| `gpqa`         | 198 | Graduate-level science QA |
| `formal_logic` | 0 (full) | Deductive reasoning (MMLU Formal Logic) |
| `hh_rlhf`      | 300 | Preference classification |

Datasets are loaded automatically via `data/data_utils.py` — just pass the name to
`--data`. Additional benchmarks (`gsm8k`, `csqa`, `minerva`, `aime24/25`, `amc23`,
`hellaswag`, `pro_medicine`, ...) are also supported.

---

## <a name="structure"></a> 📂 Project Structure

```text
UAB/
├── scripts/           # Reproduction scripts for every paper experiment
│   ├── validate.sh
│   ├── run_main_table.sh
│   ├── run_scaling.sh
│   ├── run_blackbox.sh
│   └── run_ablations.sh
└── src/
    ├── main.py        # Single entry point — all experiments
    ├── evaluator.py   # Answer parsing and metric evaluation
    ├── utils.py       # Budget-allocation algorithms (marginal-greedy, baselines)
    ├── model/         # vLLM backend and sampling configuration
    └── data/          # Benchmark dataset loaders
```

> 📁 The `out/` directory (`out/history/` for JSONL runs, plus the TSV summaries)
> is **created automatically** on the first run — you do not need to make it yourself.

---

## Acknowledgements

* [vLLM](https://github.com/vllm-project/vllm) — fast batched inference
* [Cohere](https://cohere.com/) — black-box API evaluation

This project is released under the [MIT License](LICENSE).
