import os
import json
import random
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from openai import OpenAI

MODEL = "gpt-4o-mini"


def create_openai_client(api_key: str):
    return OpenAI(api_key=api_key)


def generate_persona(agreeableness: float, job_satisfaction: float) -> str:

    text = f"You are a person with following agreeableness and job satisfaction scores: " \
           f"Your agreeableness score is {agreeableness:.2f} out of 5 and your job satisfaction score is {job_satisfaction:.2f} out of 5."

    return text


def ask_question(client: OpenAI, persona: str, question: str) -> Dict[str, float]:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": persona},
            {"role": "user", "content": question}
        ],
        temperature=1,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={"type": "json_object"}
    )

    response_text = response.choices[0].message.content
    json_response = json.loads(response_text)
    return json_response


def generate_population_responses(client: OpenAI,
                                  question: str,
                                  grid_points: List[Tuple[float, float]],
                                  n_samples: int) -> List[Dict[str, float]]:
    responses = []

    for agreeableness, job_satisfaction in grid_points:
        persona = generate_persona(agreeableness, job_satisfaction)

        for _ in range(n_samples):
            print(
                f"Generating response for agreeableness: {agreeableness}, job_satisfaction: {job_satisfaction}")
            response = ask_question(client, persona, question)
            response.update({
                "agreeableness": agreeableness,
                "job_satisfaction": job_satisfaction
            })
            responses.append(response)

    return responses


# -------------------------------------------
# Run the simulation
# -------------------------------------------
api_key = os.getenv("OPENAI_API_KEY")
client = create_openai_client(api_key)

# Create Grid Points for low low to high high
grid_points = [
    (1, 1), (1, 3), (1, 5),
    (3, 1), (3, 3), (3, 5),
    (5, 1), (5, 3), (5, 5)
]

question = """Considering your personality and other background information, how would you rate your individual performance and conflict-handling skills?
Please rate each on a scale of 1 to 5. Return your response in the following format:
reasoning: text
individual_performance: score,
conflict_handling_skills: score
as JSON
"""

# Get Responses
responses = generate_population_responses(client, question, grid_points, n_samples=10)

# Convert to DataFrame
df = pd.DataFrame(responses)
df_original = df.copy()


## -------------------------------------------
# Estimate Correlation with Bootstrapping
## -------------------------------------------

estimated_cor_matrix = []
for i in range(1000):
    df = df_original.copy()
    # -------------------------------------------
    # Add random noise all columns for numeric columns
    # -------------------------------------------
    numeric_columns = df.select_dtypes(include=['number']).columns
    df[numeric_columns] += np.random.normal(0, 1, df[numeric_columns].shape)
    cor_matrix = df[numeric_columns].corr()
    estimated_cor_matrix.append(cor_matrix)


# Get the average correlation matrix NumPy
estimated_cor_matrix = np.array(estimated_cor_matrix)
estimated_cor_matrix = np.mean(estimated_cor_matrix, axis=0)
estimated_cor_matrix = pd.DataFrame(estimated_cor_matrix)
estimated_cor_matrix.columns = numeric_columns
estimated_cor_matrix.index = numeric_columns

estimated_cor_matrix

# https://shiny.metabus.org/shiny
# 20072 | Job Satisfaction
# 20440 | Agreeableness
