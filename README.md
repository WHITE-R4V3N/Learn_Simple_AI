# Description
This is my attempt at creating simple AI models using python3. There are two simple files for two different kinds of models. One is a multi-task model that randomly generates a task based on what "user prompt" (this can be changed to actually get user input).

Another goal is to combine both the multi-tasking and data prediction models. This will mean the user can get multiple outputs. We will need to change the input data. For this we will use a input of size 5 (lets say its a 5 word corpus being used) and expect an output size of 3 (the number of tasks).

Input 1:
[0, 1, 1, 0, 0]

We expect the output to be [1, 0, 0] which will write a poem.

Input 2:
[1, 0, 0, 1, 0]

We expect the output to be [0, 1, 0] which will write a news article.

Input 2:
[0, 0, 0, 0, 1]

We expect the output to be [0, 0, 1] which will translate the sentence.

# Goal
The goal is to get this to function to a degree that all that would be needed is a LLM or a tokenizer to expand on this idea.
