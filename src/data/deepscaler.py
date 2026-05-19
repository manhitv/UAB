from datasets import load_dataset
import pandas as pd

def load_data(args, split='train'):

    dataset = load_dataset("lime-nlp/DeepScaleR_Difficulty", "Difficulty Score")['train']
    dataset = pd.DataFrame(dataset)
    
    # Only extract non fractional answers
    dataset = dataset[dataset['ground_truth'].apply(lambda x: x.isnumeric())].reset_index(drop=True)
    
    dataset = dataset.sample(frac=1, random_state=0).reset_index(drop=True).head(args.data_size)

    questions, labels, difficulties = [], [], []
    for question, answer, solved_percentage in zip(dataset['problem'], dataset['ground_truth'], dataset['solved_percentage']) :
        questions.append(question)
        labels.append(float(answer))
        
        difficulty = 1.0 - solved_percentage / 100.0
        difficulties.append(difficulty)

    return questions, labels, difficulties