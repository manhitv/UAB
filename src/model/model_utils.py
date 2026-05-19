import os
import numpy as np
# Gemma 2 uses tanh softcapping which the FA3 flash attention build does not support.
# TRITON_ATTN handles it correctly and must be set before vllm is imported.
os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN_VLLM_V1"


model_dirs = {
    'llama3.1-8b': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'llama3.2-3b': 'meta-llama/Llama-3.2-3B-Instruct',
    'llama3.2-1b': 'meta-llama/Llama-3.2-1B-Instruct',
    "gemma3-1b": "google/gemma-3-1b-it",
    "gemma3-4b": "google/gemma-3-4b-it",
    "gemma2-2b": "google/gemma-2-2b-it",
    "gemma2-9b": "google/gemma-2-9b-it",
    "falcon3-1b": "tiiuae/falcon3-1b-instruct",
    "falcon3-7b": "tiiuae/falcon3-7b-instruct",
    "falcon3-10b": "tiiuae/falcon3-10b-instruct",
    'qwen2.5-0.5b': 'Qwen/Qwen2.5-0.5B-Instruct',
    'qwen2.5-1.5b': 'Qwen/Qwen2.5-1.5B-Instruct',
    'qwen2.5-3b': 'Qwen/Qwen2.5-3B-Instruct',
    'qwen2.5-7b': 'Qwen/Qwen2.5-7B-Instruct',
    'qwen2.5-14b': 'Qwen/Qwen2.5-14B-Instruct',
    'qwen2.5-32b': 'Qwen/Qwen2.5-32B-Instruct',
    'gptoss': 'openai/gpt-oss-20b',
    'gemma3-12b': 'google/gemma-3-12b-it',
    'gemma3-27b': 'google/gemma-3-27B-it',
    'gemma4-27b': 'google/gemma-4-26B-A4B-it',
    'qwen3.6-27b': 'Qwen/Qwen3.6-27B',
    'qwen3.5-27b': 'Qwen/Qwen3.5-27B',
    'qwen3.5-9b': 'Qwen/Qwen3.5-9B',
    'qwen3-14b': 'Qwen/Qwen3-14B'
}

import re
import transformers
transformers.utils.logging.set_verbosity_error()
from vllm import SamplingParams
from model.vllm import vLLM

def _get_dtype(model_key: str, model_dir: str) -> str:
    """Return 'bfloat16' for models >=20B, 'auto' otherwise."""
    for s in [model_key, model_dir]:
        m = re.search(r'(\d+(?:\.\d+)?)b', s.lower())
        if m:
            size_b = float(m.group(1))
            return "bfloat16" if size_b >= 20 else "auto"
    return "auto"

def engine_vllm_batch(messages, agent, stop_sequences=None):

    # 1. Format prompts
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

    # 3. Generate using vLLM (Batch processing all messages at once)
    outputs = agent.llm.generate(prompts, sampling_params, use_tqdm=False)

    responses = []
    nll_scores = []
    token_stats = []

    # 4. Calculate NLL and ANLL natively from vLLM outputs
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
        min_token_nll = float(max(token_nlls)) if num_tokens > 0 else 0.0  # max NLL = least confident token

        responses.append(text)
        nll_scores.append((anll, nll, token_var, min_token_nll))
        
        # Token stats
        n_tokens = len(token_ids)
        token_stats.append({
            "input_tokens": len(output.prompt_token_ids),
            "output_tokens": n_tokens,
            "total_tokens": len(output.prompt_token_ids) + n_tokens
        })
    
    return responses, nll_scores, token_stats


def engine_vllm_multinomial(grouped_messages, agent, stop_sequences=None):
    """
    Efficient multi-sample generation using vLLM's n>1 SamplingParams.

    Each unique prompt is encoded once; n samples are drawn from a single
    forward pass — avoiding the N-fold redundant prompt encoding that occurs
    when the same question is repeated N times in engine_vllm_batch.

    Args:
        grouped_messages: list of (message, n) pairs where message is a list
                          of chat dicts and n is the number of samples needed.
    Returns:
        responses, nll_scores, token_stats — flat lists in grouped order,
        equivalent to calling engine_vllm_batch with each message repeated n times.
    """
    if not grouped_messages:
        return [], [], []

    prompts, sp_list = [], []
    for msg, n in grouped_messages:
        if isinstance(msg, list):
            prompt = agent.tokenizer.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True)
        else:
            prompt = agent.tokenizer.apply_chat_template(
                [msg], tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
        sp_list.append(SamplingParams(
            n=int(n),
            temperature=1.0,
            top_p=0.9,
            max_tokens=1024,
            stop=stop_sequences,
            logprobs=1,
        ))

    outputs = agent.llm.generate(prompts, sp_list, use_tqdm=False)

    responses, nll_scores, token_stats = [], [], []
    for output in outputs:
        for gen in output.outputs:
            token_ids      = gen.token_ids
            logprobs_list  = gen.logprobs
            token_nlls     = [-logprobs_list[i][tid].logprob
                               for i, tid in enumerate(token_ids)]
            num_tokens     = len(token_ids)
            nll            = float(sum(token_nlls))
            anll           = nll / num_tokens if num_tokens > 0 else 0.0
            token_var      = float(np.var(token_nlls)) if num_tokens > 1 else 0.0
            min_token_nll  = float(max(token_nlls))    if num_tokens > 0 else 0.0

            responses.append(gen.text.strip())
            nll_scores.append((anll, nll, token_var, min_token_nll))
            token_stats.append({
                "input_tokens":  len(output.prompt_token_ids),
                "output_tokens": num_tokens,
                "total_tokens":  len(output.prompt_token_ids) + num_tokens,
            })

    return responses, nll_scores, token_stats


def get_agents(args):

    dtype = _get_dtype(args.model, model_dirs[args.model])
    print(f"[vLLM] model={args.model}  dtype={dtype}")
    agent = vLLM(args, model_dirs[args.model], dtype=dtype)
        
    # update pad token
    if agent.tokenizer.pad_token is None :
        agent.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Personas: taken from DyLAN: https://arxiv.org/pdf/2310.02170
    if args.multi_persona :
        personas = {
            "None": "",
            "Assistant": "You are a super-intelligent AI assistant capable of performing tasks more effectively than humans.",
            "Mathematician": "You are a mathematician. You are good at math games, arithmetic calculation, and long-term planning.",
            "Economist": "You are an economist. You are good at economics, finance, and business. You have experience on understanding charts while interpreting the macroeconomic environment prevailing across world economies.",
            "Psychologist": "You are a psychologist. You are good at psychology, sociology, and philosophy. You give people scientific suggestions that will make them feel better.",
            "Lawyer": "You are a lawyer. You are good at law, politics, and history.",
            "Doctor": "You are a doctor and come up with creative treatments for illnesses or diseases. You are able to recommend conventional medicines, herbal remedies and other natural alternatives. You also consider the patient’s age, lifestyle and medical history when providing your recommendations.",
            "Programmer": "You are a programmer. You are good at computer science, engineering, and physics. You have experience in designing and developing computer software and hardware.",
            "Historian": "You are a historian. You research and analyze cultural, economic, political, and social events in the past, collect data from primary sources and use it to develop theories about what happened during various periods of history.",
            "PythonAssistant": "You are a Python writing assistant, an AI that only responds with python code, NOT ENGLISH. You will be given a function signature and its docstring by the user. Write your full implementation (restate the function signature).", # from https://github.com/composable-models/llm_multiagent_debate.git
            "AlgorithmDeveloper": "You are an algorithm developer. You are good at developing and utilizing algorithms to solve problems. You must respond with python code, no free-flowing text (unless in a comment). You will be given a function signature and its docstring by the user. Write your full implementation following the format (restate the function signature).",
            "ComputerScientist": "You are a computer scientist. You are good at writing high performance code and recognizing corner cases while solve real problems. You must respond with python code, no free-flowing text (unless in a comment). You will be given a function signature and its docstring by the user. Write your full implementation following the format (restate the function signature).",
            "CodingArtist": "You are a coding artist. You write Python code that is not only functional but also aesthetically pleasing and creative. Your goal is to make the code an art form while maintaining its utility. You will be given a function signature and its docstring by the user. Write your full implementation following the format (restate the function signature).",
            "SoftwareArchitect": "You are a software architect, skilled in designing and structuring code for scalability, maintainability, and robustness. Your responses should focus on best practices in software design. You will be given a function signature and its docstring by the user. Write your full implementation following the format (restate the function signature)."
        }
        if args.data in ['arithmetics','gsm8k']:
            personas = {
                "Assistant": "You are a super-intelligent AI assistant capable of performing tasks more effectively than humans.",
                "Mathematician": "You are a mathematician. You are good at math games, arithmetic calculation, and long-term planning.",
                "Lawyer": "You are a lawyer. You are good at law, politics, and history.",
                "Economist": "You are an economist. You are good at economics, finance, and business. You have experience on understanding charts while interpreting the macroeconomic environment prevailing across world economies.",
                "Programmer": "You are a programmer. You are good at computer science, engineering, and physics. You have experience in designing and developing computer software and hardware."
            }
        elif args.data in ['pro_medicine']:
            personas = {
                "Assistant": "You are a super-intelligent AI assistant capable of performing tasks more effectively than humans.",
                "Mathematician": "You are a mathematician. You are good at math games, arithmetic calculation, and long-term planning.",
                "Programmer": "You are a programmer. You are good at computer science, engineering, and physics. You have experience in designing and developing computer software and hardware.",
                "Psychologist": "You are a psychologist. You are good at psychology, sociology, and philosophy. You give people scientific suggestions that will make them feel better.",
                "Doctor": "You are a doctor and come up with creative treatments for illnesses or diseases. You are able to recommend conventional medicines, herbal remedies and other natural alternatives. You also consider the patient’s age, lifestyle and medical history when providing your recommendations."
            }

    else:
        personas = {"None": ""}

            
    return agent, personas
