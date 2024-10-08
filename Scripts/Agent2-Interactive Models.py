""" 
Research Question: What is the impact of assertiveness of supervisor on self-efficacy of employees?
Generate a two agent interaction that will be used to answer this research question.

Agent 1: Alex, Assistant Professor at a University
Agent 2: Matt, Assistant Professor at a University

Alex is more assertive than Matt, and Matt is more agreeable than Alex.
They will work on various project and they will discuss on how to improve the project.
"""

import os
from openai import OpenAI
from typing import List, Dict
import anthropic

MODEL = "claude-3-5-sonnet-20240620"


client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def read_text(file_path: str) -> str:
    with open(file_path, "r") as file:
        text = file.read()
    return text


def LLMCall(messages: List[Dict[str, str]], system_prompt: str = None) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        temperature=1,
        messages=messages,
        system=system_prompt
    )
    return response.content[0].text


def message_history(text: str, role: str, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    messages = messages.copy()
    messages.append({"role": role, "content": text})
    return messages


def update_memory(agent_dict: Dict, conversation_history: str, context: str) -> str:
    system_prompt = read_text("prompts/memory.txt")
    system_prompt = system_prompt.strip()
    system_prompt = system_prompt.format(previous_memory=agent_dict["memory"],
                                         name=agent_dict["name"],
                                         conversation_history=conversation_history,
                                         context=context,
                                         personality=agent_dict["personality"],
                                         attitude=agent_dict["attitude"])

    updated_memory = LLMCall([{"role": "user", "content": system_prompt}], "")
    updated_memory = updated_memory.replace(
        "<updated_memory>", "").replace("</updated_memory>", "")
    updated_memory = updated_memory.strip()
    return updated_memory


def process_response(response: str) -> str:
    # Text extract the response between <response> and </response>
    import re
    match = re.search(r"<response>(.*?)</response>", response, re.DOTALL)
    if match:
        response = match.group(1)
    else:
        response = response.strip()
    return response


def generate_response(conversation_history: str, agent_dict: Dict, context: str) -> str:

    system_prompt = read_text("prompts/generate_response.txt")

    system_prompt = system_prompt.strip()
    system_prompt = system_prompt.format(name=agent_dict["name"],
                                         personality=agent_dict["personality"],
                                         attitude=agent_dict["attitude"],
                                         memory=agent_dict["memory"],
                                         context=context,
                                         conversation=conversation_history)

    messages_process = [{"role": "user", "content": system_prompt}]
    response = LLMCall(messages_process, "")
    response = process_response(response)
    return response


agent1 = {
    "name": "Alex",
    "personality": "Level of Intellegence (1/10) | Level of Disagreeableness (1/10) | Level of Assertiveness (3/10) | Level of Support for Colleagues (10/10)",
    "attitude": "",
    "memory": "-This is a professional conversation, regarless of your feeling, personality, emotiational outburs, you are trying to be professional. No cursing or swearing. Always be professional.",
    "messages": []
}

agent2 = {
    "name": "Matt",
    "personality": "Level of Neuroticism (10/10) | Level of Risk Aversion (10/10)| Level of Intellegence (10/10)",
    "attitude": "",
    "memory": "-This is a professional conversation, regarless of your feeling, personality, emotiational outburs, you are trying to be professional. No cursing or swearing. Always be professional.",
    "messages": []
}


header = "-"*100


def simulate_interaction(agent1: Dict, agent2: Dict, num_days: int = 3, n_interaction: int = 5):
    context_history = []
    context_history_system = "You are a context generator for academic work interactions. Just generate background context, do not assign any tasks but define what the main goal"
    topic = "AI Detection Tool Implementation on Campus"
    prompt_in = f"Generate a brief bullet point background context on {topic} for an interaction between two people Professor Alex and Assistant Professor Matt. They have only {num_days} day and each day they have {n_interaction} interactions. Do not assume anything about the people, their personalities, attitudes, or personalities. Return only the context without adding any comments."
    context_history = message_history(prompt_in, "user", context_history)
    context = LLMCall(context_history, context_history_system)

    print(header)
    print(f"Context: \n {context}")
    print(header)
    conversation_history = "# Conversation History\n"

    for day_i in range(num_days):
        day_header = f"{header}\n# Day {day_i+1}\n"
        print(day_header)
        conversation_history += f"{day_header}The new day has started and the conversation is as follows:\n"

        for i in range(n_interaction):
            print(f"Interaction {i+1} of {n_interaction}")

            for agent in [agent1, agent2]:
                response = generate_response(conversation_history, agent, context).strip()
                conversation_history += f"\n{header}\n{agent['name']}: {response}\n"
                print(f"{header}\n{agent['name']}: {response}\n")

            if i == n_interaction - 2:  # Add system message before the last iteration
                system_message = f"{header}\n# System Message: Important Notes to Agents:\nThis is the second-to-last interaction for today. Please start wrapping up the conversation and prepare for the final interaction."
                print(system_message)
                conversation_history += f"{system_message}\n"
            elif i == n_interaction - 1:
                wrap_up = f"{header}\n# System Message: Important Notes to Agents:\nFor today's conversation, wrap the conversation. There is no interaction left. You must conclude the conversation for today."
                print(wrap_up)
                conversation_history += f"{wrap_up}\n"

        print(header)
        print("Memory Update...")
        print(header)
        agent1["memory"] = update_memory(agent1, conversation_history, context)
        agent2["memory"] = update_memory(agent2, conversation_history, context)

    return conversation_history

# -------------------------------------------------------
# Run the simulation
# -------------------------------------------------------

os.system("cls")
conversation_history = simulate_interaction(agent1, agent2, num_days=3, n_interaction=5)

os.system("cls")
print(agent1["memory"])
print(agent2["memory"])



def rate_performance(conversation_history: str, agent_dict: Dict):
    final_ratings = read_text("prompts/rating_generator.txt")
    final_ratings = final_ratings.format(conversation_history=conversation_history,
                                         personality=agent_dict["personality"],
                                         attitude=agent_dict["attitude"],
                                         memory=agent_dict["memory"],
                                         name=agent_dict["name"])
    return final_ratings


os.system("cls")
final_ratings = rate_performance(conversation_history, agent2)
rating_message = []
rating_message.append({"role": "user", "content": final_ratings})
final_ratings = LLMCall(rating_message, "")


print(final_ratings)
