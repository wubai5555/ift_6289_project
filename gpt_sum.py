import sys, os, json
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='dl19', choices=[
    'dl19', 'dl20', 'covid', 'news', 'scifact', 'touche'
])
parser.add_argument('--graded', action=argparse.BooleanOptionalAction, help="whether eval model trained on graded rel")
args = parser.parse_args()
print(args)
save_dir = f'gpt/{args.dataset}.json'
GPT_MODEL = "gpt-3.5-turbo-0125"
client = OpenAI()


@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, tools=None, tool_choice=None, model=GPT_MODEL):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

def pretty_print_conversation(messages):
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
        "function": "magenta",
    }
    
    for message in messages:
        if message["role"] == "system":
            print(colored(f"system: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "user":
            print(colored(f"user: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "assistant" and message.get("function_call"):
            print(colored(f"assistant: {message['function_call']}\n", role_to_color[message["role"]]))
        elif message["role"] == "assistant" and not message.get("function_call"):
            print(colored(f"assistant: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "function":
            print(colored(f"function ({message['name']}): {message['content']}\n", role_to_color[message["role"]]))

TEMPLATE = """
Help me to summarize the following passage: {p}
"""
MAX_LEN = 15000
with open(save_dir, 'r') as f:
    news_json = json.load(f)
for each in news_json:
    hits = each['hits']
    for hit in hits:
        if 'summary' in hit:
            continue
        else:
            content = hit['content'][:MAX_LEN]
            messages = [{'role': 'user', 'content': TEMPLATE.format(p=content)}]
            pretty_print_conversation(messages)
            response = chat_completion_request(messages=messages)
            content = response.choices[0].message.content
            print(content)
            hit['summary'] = content
            with open(save_dir, 'w') as f:
                json.dump(news_json, f, indent=4)
