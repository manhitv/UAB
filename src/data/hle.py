from datasets import load_dataset
import pandas as pd


def load_data(args, split='test'):
    # Humanity's Last Exam — text-only subset (filter out image questions)
    dataset = load_dataset('cais/hle')['test']
    dataset = pd.DataFrame(dataset)

    # Keep only questions without an image attachment
    if 'image' in dataset.columns:
        dataset = dataset[dataset['image'].isna() | (dataset['image'] == '')].reset_index(drop=True)

    dataset = dataset.sample(frac=1, random_state=0).reset_index(drop=True).head(args.data_size)

    questions, labels = [], []
    for question, answer in zip(dataset['question'], dataset['answer']):
        questions.append(question)
        labels.append(answer)

    return questions, labels
