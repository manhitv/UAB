import json
import pandas as pd

DATA_PATH = '/home/s224852302/UGCS/evaluation/data/math/{split}.jsonl'


def load_data(args, split='test'):
    """MATH Algebra subset from local UGCS data — Level 1-5, difficulty mapped to [0.0, 1.0]."""
    path = DATA_PATH.format(split=split)
    with open(path) as f:
        records = [json.loads(line) for line in f if json.loads(line).get('subject') == 'Algebra']

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=0).reset_index(drop=True).head(args.data_size)

    questions, labels, difficulties = [], [], []
    for problem, answer, level in zip(df['problem'], df['answer'], df['level']):
        try:
            level_num = int(str(level).split()[-1])
            difficulty = (level_num - 1) / 4.0
        except (ValueError, IndexError):
            difficulty = 0.5

        questions.append(problem)
        labels.append(answer)
        difficulties.append(difficulty)

    return questions, labels, difficulties
