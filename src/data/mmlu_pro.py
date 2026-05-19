
from datasets import load_dataset
import pandas as pd

def load_data(args, split='test', n_sample_per_cat=10):
    # split = 'validation' | 'test'
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")[split]
    dataset = pd.DataFrame(dataset)
    
    # Filter
    selected_categories = dataset['category'].unique()
    selected_question_idx = []
    for cat in selected_categories:
        question_idx = dataset.loc[dataset['category'] == cat, 'question_id'].tolist()
        selected_question_idx.extend(question_idx[:n_sample_per_cat])
    
    dataset = dataset[dataset['question_id'].isin(selected_question_idx)]

    # Prepare questions and labels
    questions, labels = [], []
    choices = "ABCDEFGHIJ"
    
    template = '{}\n(A) {}\n(B) {}\n(C) {}\n(D) {}\n(E) {}\n(F) {}\n(G) {}\n(H) {}\n(I) {}\n(J) {}\n\n'
    for query, options, answer in zip(dataset['question'], dataset['options'], dataset['answer_index']):
        if len(options) != 10 :
            continue
        question = template.format(query, options[0], options[1], options[2], options[3], options[4], options[5], 
                                   options[6], options[7], options[8], options[9])
        label = f"({choices[int(answer)]})"
        questions.append(question)
        labels.append(label)

    return questions, labels