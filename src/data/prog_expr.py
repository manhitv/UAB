from datasets import Dataset
import random
import numpy as np
from sympy import sympify


def get_difficulty_params(level: int):
    level = max(1, min(10, level))
    num_operands = 2 + level // 2
    max_digits = 1 + min(3, (level - 1) // 3)
    ops = ['+', '-']
    if level >= 3:
        ops.append('*')
    if level >= 5:
        ops.append('/')
    use_parentheses = level >= 4
    paren_prob = 0.0 if not use_parentheses else min(0.6, 0.2 + 0.05 * (level - 4))
    use_exponent = level >= 8
    exponent_prob = 0.0 if not use_exponent else min(0.5, 0.2 * (level - 7))
    return {
        'num_operands': num_operands,
        'max_digits': max_digits,
        'operators': ops,
        'use_parentheses': use_parentheses,
        'paren_prob': paren_prob,
        'use_exponent': use_exponent,
        'exponent_prob': exponent_prob,
    }


def generate_expression(level: int, seed: int = None):
    if seed is not None:
        random.seed(seed)

    params = get_difficulty_params(level)

    numbers = []
    for _ in range(params['num_operands']):
        max_val = 10 ** params['max_digits'] - 1
        num = random.randint(1, max_val)
        numbers.append(num)

    operators_pool = params['operators']

    expr_parts = [str(numbers[0])]
    open_paren = 0

    for i in range(1, params['num_operands']):
        op = random.choice(operators_pool)
        next_num = str(numbers[i])

        if params['use_parentheses'] and random.random() < params['paren_prob'] and open_paren == 0:
            expr_parts.append(op)
            expr_parts.append("(")
            expr_parts.append(next_num)
            open_paren += 1
        else:
            expr_parts.append(op)
            expr_parts.append(next_num)

        if open_paren > 0 and random.random() < 0.6:
            expr_parts.append(")")
            open_paren -= 1

    expression = " ".join(expr_parts) + ")" * open_paren

    if params['use_exponent'] and random.random() < params['exponent_prob']:
        exp = random.randint(2, 3)
        expression = f"({expression}) ** {exp}"

    try:
        answer = float(sympify(expression))
    except Exception:
        return generate_expression(level, seed=random.randint(0, 999999))

    if not np.isfinite(answer) or abs(answer) > 1e9:
        return generate_expression(level, seed=random.randint(0, 999999))

    question = f"What is the result of {expression}?"

    return {
        'question': question,
        'answer': round(answer, 6) if answer % 1 != 0 else int(answer),
        'difficulty_level': level,
        'expression': expression,
    }


def load_data(args, split='train'):
    data_size = args.data_size
    base_seed = 42 if split == 'train' else 43

    X, Y, difficulties = [], [], []

    for i in range(data_size):
        if split == 'train':
            level = random.randint(1, 10)
        else:
            level = 1 + (i * 10) // data_size

        sample = generate_expression(level, seed=base_seed + i)

        X.append(sample['question'])
        Y.append(sample['answer'])
        difficulties.append(sample['difficulty_level'])

    return X, Y, difficulties
