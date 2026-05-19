from datasets import load_dataset
import pandas as pd


def extract_answer(answer_str: str):
    try:
        return int(str(answer_str).strip())
    except ValueError:
        return None


def load_data(args, split='test'):
    # 30 problems from AIME 2024 I & II
    dataset = load_dataset('AI-MO/aimo-validation-aime')['train']
    dataset = pd.DataFrame(dataset)

    # Filter to 2024 problems only (dataset may contain other years)
    if 'url' in dataset.columns:
        dataset = dataset[dataset['url'].str.contains('2024', na=False)].reset_index(drop=True)

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
