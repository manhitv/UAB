from datasets import load_dataset
import pandas as pd


def load_data(args, split='test'):
    # 272 STEM problems from the MINERVA paper evaluation set (physics, chemistry, math)
    # Answers are decimal strings, e.g. '1.6'
    dataset = load_dataset('math-ai/minervamath')['test']
    dataset = pd.DataFrame(dataset)
    dataset = dataset.sample(frac=1, random_state=0).reset_index(drop=True).head(args.data_size)

    questions, labels = [], []
    for question, answer in zip(dataset['question'], dataset['answer']):
        questions.append(question)
        labels.append(answer)

    return questions, labels
