# Author:       Emma Gillespie
# Date:         2024-03-18
# Description:  A simplified Python3 and Numpy AI Model for text processing and basic task prediction.

#### Imports ####

import numpy as np

#### Code ####

# Define tasks and probabilities
tasks = ["Write a poem", "Summarize a new article", "Translate a sentence"]
probabilities = np.array([0.3, 0.5, 0.2])

# Preprocess user input (simple word count here)
def preprocess(text):
    return len(text.split())

# Predict task based on prompt length and probabilities
def predict_task(text):
    word_count = preprocess(text)
    threshold = np.random.rand()
    cumalitive_prob = 0

    for i, task in enumerate(tasks):
        cumalitive_prob += probabilities[i]
        if threshold <= cumalitive_prob:
            return task, word_count
        

# Example usage
user_prompt =  "This is a very long news article about the advancements of AI and LLM's. No matter what happens this is only random and it does not learn."
predicted_task, word_count = predict_task(user_prompt)
print(f'Predicted task: {predicted_task} based on {word_count} words.')

#### Optional ####
# You can write code to perform the tasks that the simplified AI predicts.