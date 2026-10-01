import random, re, collections
import numpy as np

def get_instruction_suffix(args):
    if args.data == 'prog_expr':
        example = "12.34"
        head = " Make sure to state your final answer in curly brackets at the very end of your response, just like: "
    elif args.data in ['gsm8k', 'deepscaler']:
        example = "123"
        head = " Make sure to state your final answer in curly brackets at the very end of your response, just like: "
    elif args.data == 'formal_logic':
        example = "(A)"
        head = " Make sure to state your final answer choice in curly brackets at the very end of your response, just like: "
    elif args.data == 'math500':
        if args.cot:
            return " Put your final answer in \\boxed{answer} at the very end of your response, e.g. \\boxed{\\frac{1}{2}} or \\boxed{3.14}. Let's think step by step."
        return ' Put your final answer in \\boxed{answer} at the very end of your response, e.g. \\boxed{\\frac{1}{2}} or \\boxed{3.14}.'
    else:
        raise NotImplementedError(f"Unknown dataset: {args.data}")
    if args.cot:
        return head + "'{final answer: " + example + "}'. Let's think step by step."
    return head + '"{final answer: ' + example + '}".'


def evaluate_arithmetics(responses, answer):
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
        debate_answer = random.choice(most_common)

    return final_answers, debate_answer, debate_answer == np.round(answer, 1)


def evaluate_mcq(responses, answer):
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
        debate_answer = random.choice(most_common)
    return final_answers, debate_answer, debate_answer == answer


def _extract_last_boxed(text):
    idx = text.rfind(r'\boxed{')
    if idx == -1:
        return None
    start = idx + 7
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
    s = str(s).strip()
    m = re.match(r'^\\boxed\{(.+)\}$', s, re.DOTALL)
    if m:
        s = m.group(1).strip()
    s = re.sub(r'\\left|\\right', '', s)
    s = s.replace(' ', '').lower()
    return s


def math_answers_equal(pred, gold, tol=1e-4):
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
    counter = collections.Counter(values)
    max_count = max(counter.values())
    return random.choice([k for k, v in counter.items() if v == max_count])


def evaluate_math_latex(responses, answer):
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
