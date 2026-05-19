from datasets import load_dataset
import pandas as pd


def load_data(args, split='test'):
    # 500 problems from the MATH benchmark, covering 5 difficulty levels
    # Returns (questions, labels, difficulties) like deepscaler — Level 1-5 mapped to [0.0, 1.0]
    dataset = load_dataset('HuggingFaceH4/MATH-500')['test']
    dataset = pd.DataFrame(dataset)
    dataset = dataset.sample(frac=1, random_state=0).reset_index(drop=True).head(args.data_size)

    questions, labels, difficulties = [], [], []
    for problem, answer, level in zip(dataset['problem'], dataset['answer'], dataset['level']):
        try:
            # level format: "Level 3"
            level_num = int(str(level).split()[-1])
            difficulty = level_num # (level_num - 1) / 4.0
        except (ValueError, IndexError):
            difficulty = 3 # 0.5

        questions.append(problem)
        labels.append(answer)
        difficulties.append(difficulty)

    return questions, labels, difficulties
