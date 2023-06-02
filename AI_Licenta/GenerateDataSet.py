import pandas as pd
import random
import numpy as np


def generateDataSet(num_samples, num_questions):
    # Create an empty list to store the data

    generatedCases = []
    for _ in range(5):
        generatedCases.append([0, 0])

    data = []
    all_scores = []

    # Generate random question scores between 1 and 5 for each sample
    for _ in range(num_samples):
        caseGenerated = False
        while not caseGenerated:
            sample_data = []
            for i in range(num_questions):
                score = random.randint(1, 5)
                sample_data.append(score)
            scores = [0 for _ in range(5)]
            scores[0] = calculate_for_nutrition(sample_data)
            scores[1] = calculate_for_anxiety(sample_data)
            scores[2] = calculate_for_communication(sample_data)
            scores[3] = calculate_for_concentration(sample_data)
            scores[4] = calculate_for_sport(sample_data)
            scores = normalize_scores(scores)

            caseGenerated = True
            for i in range(5):
                if generatedCases[i][scores[i]] >= num_samples // 2:
                    caseGenerated = False
                    break
            if caseGenerated:
                all_scores.append(scores)
                data.append(sample_data)
                for i in range(5):
                    generatedCases[i][scores[i]] += 1

    # Create a DataFrame from the data list
    df = pd.DataFrame(np.column_stack([data, all_scores]), columns=[f"Q{i}" for i in range(1, num_questions + 1)] +
                                                                   [f"S{i}" for i in range(1, 6)])

    # Save the dataset to a CSV file
    df.to_csv('question_scores_dataset.csv', index=False)


def calculate_for_nutrition(answers):
    relevant = answers[0:3]
    return 0.1 * relevant[0] + 0.2 * relevant[2] - 0.1 * relevant[1]


def calculate_for_anxiety(answers):
    relevant = answers[3:6]
    return 0.1 * relevant[0] + 0.05 * relevant[1] + 0.05 * relevant[2]


def calculate_for_communication(answers):
    relevant = answers[6:9]
    return 0.1 * relevant[0] + 0.05 * relevant[1] + 0.05 * relevant[2]


def calculate_for_concentration(answers):
    relevant = answers[9:12]
    return 0.1 * relevant[0] + 0.05 * relevant[1] + 0.05 * relevant[2]


def calculate_for_sport(answers):
    relevant = answers[12:15]
    return 1 - 0.05 * relevant[0] - 0.1 * relevant[1] - 0.05 * relevant[2]


def normalize_scores(scores):
    for i in range(len(scores)):
        if scores[i] < 0.45:
            scores[i] = 0
        elif 0.45 <= scores[i] <= 0.55:
            result = random.randint(0, 1)
            scores[i] = result
        else:
            scores[i] = 1

    return scores
