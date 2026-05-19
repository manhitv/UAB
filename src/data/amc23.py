from datasets import load_dataset
import pandas as pd


def extract_answer(answer_str: str):
    # AMC answers are integers (AMC 10/12: 0-999, answers are one of 5 choices A-E mapped to values)
    answer_str = str(answer_str).strip()
    try:
        return int(answer_str)
    except ValueError:
        return answer_str if answer_str else None


def load_data(args, split='test'):
    # AMC 2023 competition problems (AMC 8, 10, 12)
    dataset = load_dataset('AI-MO/aimo-validation-amc')['train']
    dataset = pd.DataFrame(dataset)

    # Filter to 2023 problems via URL
    if 'url' in dataset.columns:
        dataset = dataset[dataset['url'].str.contains('2023', na=False)].reset_index(drop=True)

    dataset = dataset.sample(frac=1, random_state=0).reset_index(drop=True)
    if split != 'train':
        dataset = dataset.head(args.data_size)

    questions, labels = [], []
    for problem, answer in zip(dataset['problem'], dataset['answer']):
        label = extract_answer(answer)
        if label is not None:
            questions.append(problem)
            labels.append(label)

    return questions, labels
