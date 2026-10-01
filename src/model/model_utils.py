import os
os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN_VLLM_V1"

import re
import numpy as np
import transformers
from vllm import SamplingParams
from model.vllm import vLLM

transformers.utils.logging.set_verbosity_error()

model_dirs = {
    'qwen2.5-1.5b': 'Qwen/Qwen2.5-1.5B-Instruct',
    'qwen2.5-7b': 'Qwen/Qwen2.5-7B-Instruct',
    'llama3.2-3b': 'meta-llama/Llama-3.2-3B-Instruct',
    'gptoss': 'openai/gpt-oss-20b',
    'gemma3-27b': 'google/gemma-3-27B-it',
}


def _get_dtype(model_key: str, model_dir: str) -> str:
    for s in [model_key, model_dir]:
        m = re.search(r'(\d+(?:\.\d+)?)b', s.lower())
        if m:
            size_b = float(m.group(1))
            return "bfloat16" if size_b >= 20 else "auto"
    return "auto"


def engine_vllm_batch(messages, agent, stop_sequences=None):

    if type(messages[0]) == list:
        prompts = [agent.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) for msgs in messages]
    else:
        prompts = [agent.tokenizer.apply_chat_template([msg], tokenize=False, add_generation_prompt=True) for msg in messages]

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.9,
        max_tokens=1024,
        stop=stop_sequences,
        logprobs=1
    )

    outputs = agent.llm.generate(prompts, sampling_params, use_tqdm=False)

    responses = []
    nll_scores = []
    token_stats = []

    for output in outputs:
        generated_sequence = output.outputs[0]
        text = generated_sequence.text.strip()
        token_ids = generated_sequence.token_ids
        logprobs_list = generated_sequence.logprobs

        token_nlls = []
        for i, token_id in enumerate(token_ids):
            token_nlls.append(-logprobs_list[i][token_id].logprob)

        num_tokens = len(token_ids)
        nll = float(sum(token_nlls))
        anll = nll / num_tokens if num_tokens > 0 else 0.0
        token_var = float(np.var(token_nlls)) if num_tokens > 1 else 0.0
        min_token_nll = float(max(token_nlls)) if num_tokens > 0 else 0.0

        responses.append(text)
        nll_scores.append((anll, nll, token_var, min_token_nll))

        n_tokens = len(token_ids)
        token_stats.append({
            "input_tokens": len(output.prompt_token_ids),
            "output_tokens": n_tokens,
            "total_tokens": len(output.prompt_token_ids) + n_tokens
        })

    return responses, nll_scores, token_stats


def get_agents(args):
    dtype = _get_dtype(args.model, model_dirs[args.model])
    print(f"[vLLM] model={args.model}  dtype={dtype}")
    agent = vLLM(args, model_dirs[args.model], dtype=dtype)
    if agent.tokenizer.pad_token is None:
        agent.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return agent
