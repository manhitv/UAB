from datasets import load_dataset
import pandas as pd


def extract_answer(answer_str: str):
    try:
        return int(str(answer_str).strip())
    except ValueError:
        return None


def load_data(args, split='test'):
    # 30 problems from AIME 2025 I & II
    # Alternative dataset ID: 'AI-MO/aimo-validation-aime' with 2025 URL filter
    dataset = load_dataset('math-ai/aime25')['test']
    dataset = pd.DataFrame(dataset)

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
