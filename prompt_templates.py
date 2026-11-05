"""Editable prompt templates for the four techniques (Unit I)."""

TEMPLATES = {
    "Zero-shot": "{task}",
    "Few-shot": "Here are some examples of the expected style:\n\n{examples}\n\nNow answer this task in the same style.\nTask: {task}\nAnswer:",
    "Chain-of-Thought": "Task: {task}\n\nWork through this step by step. Give a short numbered list of key steps "
                        "(concise justification only), then finish with a line starting 'Final answer:'.",
    "Role-based": "You are {role}. Answer the task below the way that expert would: accurate, clear, and appropriate "
                  "for the audience.\n\nTask: {task}",
}

EXPLANATIONS = {
    "Zero-shot": "The task is sent directly with no examples. It tests what the model already knows.",
    "Few-shot": "A few worked examples are placed before the task so the model copies their format and style.",
    "Chain-of-Thought": "The model is asked to reason in steps and end with a clear final answer. We request a brief justification, not private reasoning.",
    "Role-based": "The model is assigned an expert persona, which changes tone, vocabulary and focus.",
}

DEFAULT_ROLES = {
    "Question answering": "a knowledgeable university teacher",
    "Explanation": "a patient teacher explaining to a first-year student",
    "Summarization": "a professional editor",
    "Sentiment analysis": "a data analyst",
    "Creative writing": "an award-winning author",
    "Reasoning / maths": "a maths tutor",
}

FEW_SHOT_EXAMPLES = {
    "Question answering": [("What is overfitting?", "Overfitting is when a model learns training data too closely, including noise, so it performs poorly on new data."),
                           ("What is a GAN?", "A GAN is a pair of networks, a generator and a discriminator, trained against each other to produce realistic data.")],
    "Explanation": [("Explain gradient descent.", "Gradient descent repeatedly moves model parameters a small step in the direction that reduces the loss."),
                    ("Explain attention.", "Attention lets a model weigh which input tokens matter most when producing each output token.")],
    "Summarization": [("Summarize: The model was trained for ten epochs on a small dataset and reached 92% accuracy.", "A small-data model reached 92% accuracy after 10 epochs.")],
    "Sentiment analysis": [("The battery life is awful.", "Negative"), ("Loved the camera, great value!", "Positive")],
    "Creative writing": [("Write a one-line story about a robot.", "The robot learned to dream the night its battery finally ran out.")],
    "Reasoning / maths": [("A train goes 60 km in 1.5 hours. Speed?", "60 / 1.5 = 40 km/h. Final answer: 40 km/h"),
                          ("What is 15% of 200?", "0.15 x 200 = 30. Final answer: 30")],
}
