

import random, re, collections
import numpy as np


def get_instruction_suffix(args):
    if args.data in ['arithmetics', 'prog_expr']:
        if args.bae :
            return ' Make sure to state your answer at the end of the response.'
        elif args.cot :
            return " Make sure to state your final answer in curly brackets at the very end of your response, just like: '{final answer: 12.34}'. Let's think step by step."
        else :
            return ' Make sure to state your final answer in curly brackets at the very end of your response, just like: "{final answer: 12.34}".'
    elif args.data in ['gsm8k', 'deepscaler']:
        if args.bae :
            return ' Make sure to state your answer at the end of the response.'
        elif args.cot :
            return " Make sure to state your final answer in curly brackets at the very end of your response, just like: '{final answer: 123}'. Let's think step by step."
        else :
            return ' Make sure to state your final answer in curly brackets at the very end of your response, just like: "{final answer: 123}".'
    elif args.data in ['hellaswag','pro_medicine','formal_logic','csqa','hh_rlhf', 'mmlu_pro', 'gpqa']:
        if args.bae :
            return ' Put your final answer in the form (X) at the end of your response.'
        elif args.cot :
            return " Make sure to state your final answer choice in curly brackets at the very end of your response, just like: '{final answer: (A)}'. Let's think step by step."
        else :
            return ' Make sure to state your final answer choice in curly brackets at the very end of your response, just like: "{final answer: (A)}".'
    elif args.data in ['cnn_daily']:
        return ' Make sure to provide your summary after stating "# Summary # ".'
    elif args.data in ['aime24', 'aime25', 'amc23']:
        if args.bae :
            return ' Make sure to state your integer answer at the end of the response.'
        elif args.cot :
            return " Make sure to state your final integer answer in curly brackets at the very end of your response, just like: '{final answer: 123}'. Let's think step by step."
        else :
            return ' Make sure to state your final integer answer in curly brackets at the very end of your response, just like: "{final answer: 123}".'
    elif args.data in ['math500', 'minerva', 'math_algebra']:
        if args.bae :
            return ' Make sure to put your final answer in \\boxed{} at the end of your response.'
        elif args.cot :
            return " Put your final answer in \\boxed{answer} at the very end of your response, e.g. \\boxed{\\frac{1}{2}} or \\boxed{3.14}. Let's think step by step."
        else :
            return ' Put your final answer in \\boxed{answer} at the very end of your response, e.g. \\boxed{\\frac{1}{2}} or \\boxed{3.14}.'
    elif args.data in ['hle']:
        if args.bae :
            return ' Make sure to state your answer at the end of the response.'
        elif args.cot :
            return " Make sure to state your final answer in curly brackets at the very end of your response, just like: '{final answer: X}'. Let's think step by step."
        else :
            return ' Make sure to state your final answer in curly brackets at the very end of your response, just like: "{final answer: X}".'



def evaluate_arithmetics(responses, answer):
    # Returns True if corret, False if incorrect
    final_answers = []
    for _, response in responses.items():

        try:
            pred = re.findall(r"\{(.*?)\}", response)[-1]
            pred = float(pred.replace("final answer:", "").strip())
            final_answers.append(np.round(pred, 1))
        except :
            final_answers.append("")


    if len(set(final_answers)) == 1 and list(set(final_answers))[0] == "":
        final_answers = [""] * len(final_answers)
        debate_answer = ""
    else :
        counter = collections.Counter([x for x in final_answers if x != ""])
        max_count = max(counter.values())
        most_common = [key for key, value in counter.items() if value == max_count]
        debate_answer = random.choice(most_common) # if there is a tie, will choose randomly

    return final_answers, debate_answer, debate_answer == np.round(answer, 1)


def evaluate_mcq(responses, answer):
    # Returns True if corret, False if incorrect
    final_answers = []
    for _, response in responses.items():

        try:
            pred = re.findall(r"\{(.*?)\}", response)[-1]
            pred = pred.replace("final answer:", "").strip()
            if len(pred) == 0 :
                final_answers.append("")
            elif len(pred) < 3 :
                pred = pred[0]
                final_answers.append(f"({pred})")
            else :
                pred = pred[1]
                final_answers.append(f"({pred})")
        except :
            final_answers.append("")
    
    if len(set(final_answers)) == 1 and list(set(final_answers))[0] == "":
        final_answers = [""] * len(final_answers)
        debate_answer = ""
    else :
        counter = collections.Counter([x for x in final_answers if x != ""])
        max_count = max(counter.values())
        most_common = [key for key, value in counter.items() if value == max_count]
        debate_answer = random.choice(most_common) # if there is a tie, will choose randomly
    return final_answers, debate_answer, debate_answer == answer

def base_evaluate_arithmetics(responses, answer):
    final_answers = []
    for _, sentence in responses.items():
        parts = sentence.split(" ")

        for part in parts[::-1]:
            try:
                ans = float(part)
                final_answers.append(ans)
                break
            except:
                continue

    counter = collections.Counter([x for x in final_answers if x != ""])
    try:
        max_count = max(counter.values())
        most_common = [key for key, value in counter.items() if value == max_count]
        debate_answer = random.choice(most_common) # if there is a tie, will choose randomly
    except :
        debate_answer = "" 

    return final_answers, debate_answer, debate_answer == np.round(answer, 1)

def base_evaluate_mcq(responses, answer):

    final_answers = []
    for _, input_str in responses.items():

        pattern = r'\((\w)\)'
        matches = re.findall(pattern, input_str)

        solution = None
        for match_str in matches[::-1]:
            solution = match_str.upper()
            if solution:
                final_answers.append(f"({solution})")
                break

    counter = collections.Counter([x for x in final_answers if x != ""])
    try :
        max_count = max(counter.values())
        most_common = [key for key, value in counter.items() if value == max_count]
        debate_answer = random.choice(most_common) # if there is a tie, will choose randomly
    except :
        debate_answer = ""
    return final_answers, debate_answer, debate_answer == answer


# ── Math helpers ──────────────────────────────────────────────────────────────

def _extract_last_boxed(text):
    """Extract content of the last \\boxed{...}, handling nested braces."""
    idx = text.rfind(r'\boxed{')
    if idx == -1:
        return None
    start = idx + 7  # skip past \boxed{
    depth = 1
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
    return None


def _eval_frac(s):
    """Parse \\frac{a}{b}, -\\frac{a}{b}, or plain float to a Python float."""
    s = s.strip()
    m = re.match(r'^(-?)\\?frac\{([^}]+)\}\{([^}]+)\}$', s)
    if m:
        sign = -1 if m.group(1) else 1
        try:
            return sign * float(m.group(2)) / float(m.group(3))
        except (ValueError, ZeroDivisionError):
            pass
    try:
        return float(s)
    except ValueError:
        return None


def _normalize_latex(s):
    """Normalize a LaTeX math string for string comparison."""
    s = str(s).strip()
    # Unwrap \boxed{...}
    m = re.match(r'^\\boxed\{(.+)\}$', s, re.DOTALL)
    if m:
        s = m.group(1).strip()
    s = re.sub(r'\\left|\\right', '', s)
    s = s.replace(' ', '').lower()
    return s


def math_answers_equal(pred, gold, tol=1e-4):
    """Return True if pred matches gold — exact string or numeric (LaTeX-aware)."""
    if not pred:
        return False
    p, g = _normalize_latex(pred), _normalize_latex(str(gold))
    if p == g:
        return True
    pv, gv = _eval_frac(p), _eval_frac(g)
    if pv is not None and gv is not None:
        return abs(pv - gv) <= tol
    return False


def _majority(values):
    """Return majority element from a non-empty list; break ties randomly."""
    counter = collections.Counter(values)
    max_count = max(counter.values())
    return random.choice([k for k, v in counter.items() if v == max_count])


# ── Evaluate: math competition (AIME24, AIME25, AMC23) — integer answers ─────

def evaluate_math_competition(responses, answer):
    final_answers = []
    for _, response in responses.items():
        try:
            pred = re.findall(r"\{(.*?)\}", response)[-1]
            pred = int(pred.replace("final answer:", "").strip())
            final_answers.append(pred)
        except:
            final_answers.append("")

    non_empty = [x for x in final_answers if x != ""]
    debate_answer = _majority(non_empty) if non_empty else ""
    try:
        correct = int(debate_answer) == int(answer)
    except:
        correct = False
    return final_answers, debate_answer, correct


def base_evaluate_math_competition(responses, answer):
    """BAE fallback: scan for the last integer in the response."""
    final_answers = []
    for _, response in responses.items():
        numbers = re.findall(r'\b\d{1,3}\b', response)
        try:
            final_answers.append(int(numbers[-1]))
        except:
            final_answers.append("")

    non_empty = [x for x in final_answers if x != ""]
    debate_answer = _majority(non_empty) if non_empty else ""
    try:
        correct = int(debate_answer) == int(answer)
    except:
        correct = False
    return final_answers, debate_answer, correct


# ── Evaluate: math open (MATH-500, MINERVA) — LaTeX / decimal answers ─────────

def evaluate_math_latex(responses, answer):
    """Extract \\boxed{} answer; fall back to {final answer: ...} format."""
    final_answers = []
    for _, response in responses.items():
        pred = _extract_last_boxed(response)
        if pred is None:
            try:
                pred = re.findall(r"\{(.*?)\}", response)[-1]
                pred = pred.replace("final answer:", "").strip()
            except:
                pred = ""
        final_answers.append(pred if pred else "")

    non_empty = [x for x in final_answers if x != ""]
    if not non_empty:
        return final_answers, "", False

    norm_to_orig = {}
    for x in non_empty:
        norm_to_orig.setdefault(_normalize_latex(x), x)

    debate_norm = _majority([_normalize_latex(x) for x in non_empty])
    debate_answer = norm_to_orig[debate_norm]
    return final_answers, debate_answer, math_answers_equal(debate_answer, answer)


def base_evaluate_math_latex(responses, answer):
    """BAE fallback: look for \\boxed{} or last number-like token."""
    final_answers = []
    for _, response in responses.items():
        pred = _extract_last_boxed(response)
        if pred is None:
            nums = re.findall(r'-?\d+\.?\d*', response)
            pred = nums[-1] if nums else ""
        final_answers.append(pred if pred else "")

    non_empty = [x for x in final_answers if x != ""]
    if not non_empty:
        return final_answers, "", False

    norm_to_orig = {}
    for x in non_empty:
        norm_to_orig.setdefault(_normalize_latex(x), x)

    debate_norm = _majority([_normalize_latex(x) for x in non_empty])
    debate_answer = norm_to_orig[debate_norm]
    return final_answers, debate_answer, math_answers_equal(debate_answer, answer)


# ── Evaluate: HLE — exact string / numeric match ──────────────────────────────

def _hle_correct(pred, gold):
    if pred.lower().strip() == str(gold).lower().strip():
        return True
    try:
        return abs(float(pred) - float(gold)) <= 1e-4
    except:
        return False


def evaluate_hle(responses, answer):
    final_answers = []
    for _, response in responses.items():
        try:
            pred = re.findall(r"\{(.*?)\}", response)[-1]
            pred = pred.replace("final answer:", "").strip()
        except:
            pred = ""
        final_answers.append(pred)

    non_empty = [x for x in final_answers if x != ""]
    if not non_empty:
        return final_answers, "", False

    debate_answer = _majority([x.lower().strip() for x in non_empty])
    # recover original casing
    debate_answer = next(x for x in non_empty if x.lower().strip() == debate_answer)
    return final_answers, debate_answer, _hle_correct(debate_answer, answer)


def base_evaluate_hle(responses, answer):
    """BAE fallback: check \\boxed{} then last non-empty line."""
    final_answers = []
    for _, response in responses.items():
        pred = _extract_last_boxed(response)
        if pred is None:
            lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
            pred = lines[-1] if lines else ""
        final_answers.append(pred if pred else "")

    non_empty = [x for x in final_answers if x != ""]
    if not non_empty:
        return final_answers, "", False

    debate_answer = _majority([x.lower().strip() for x in non_empty])
    debate_answer = next(x for x in non_empty if x.lower().strip() == debate_answer)
    return final_answers, debate_answer, _hle_correct(debate_answer, answer)




