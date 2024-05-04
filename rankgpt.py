import copy
from tqdm import tqdm
import time
import json
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored
import tempfile
from typing import Dict, List
from benchmark import THE_TOPICS
from trec_eval import EvalFunction

GPT_MODEL = "gpt-3.5-turbo-0125"
client = OpenAI()
@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, tools=None, tool_choice=None, temperature=None, model=GPT_MODEL):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

def get_prefix_prompt(query, num):
    return [{'role': 'system',
             'content': "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."},
            {'role': 'user',
             'content': f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}."},
            {'role': 'assistant', 'content': 'Okay, please provide the passages.'}]


def get_post_prompt(query, num):
    return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain."


def create_permutation_instruction(item=None, rank_start=0, rank_end=100):
    query = item['query']
    num = len(item['hits'][rank_start: rank_end])

    max_length = 300

    messages = get_prefix_prompt(query, num)
    rank = 0
    for hit in item['hits'][rank_start: rank_end]:
        rank += 1
        # content = hit['summary']
        content = hit['content']
        # content = content.replace('Title: Content: ', '')
        content = content.strip()
        # For Japanese should cut by character: content = content[:int(max_length)]
        content = ' '.join(content.split()[:int(max_length)])
        messages.append({'role': 'user', 'content': f"[{rank}] {content}"})
        messages.append({'role': 'assistant', 'content': f'Received passage [{rank}].'})
    messages.append({'role': 'user', 'content': get_post_prompt(query, num)})

    return messages


def run_llm(messages):
    response = chat_completion_request(messages=messages, temperature=0)
    return response.choices[0].message.content


def clean_response(response: str):
    new_response = ''
    for c in response:
        if not c.isdigit():
            new_response += ' '
        else:
            new_response += c
    new_response = new_response.strip()
    return new_response


def remove_duplicate(response):
    new_response = []
    for c in response:
        if c not in new_response:
            new_response.append(c)
    return new_response


def receive_permutation(item, permutation, rank_start=0, rank_end=100):
    response = clean_response(permutation)
    response = [int(x) - 1 for x in response.split()]
    response = remove_duplicate(response)
    cut_range = copy.deepcopy(item['hits'][rank_start: rank_end])
    original_rank = [tt for tt in range(len(cut_range))]
    response = [ss for ss in response if ss in original_rank]
    response = response + [tt for tt in original_rank if tt not in response]
    for j, x in enumerate(response):
        item['hits'][j + rank_start] = copy.deepcopy(cut_range[x])
        if 'rank' in item['hits'][j + rank_start]:
            item['hits'][j + rank_start]['rank'] = cut_range[j]['rank']
        if 'score' in item['hits'][j + rank_start]:
            item['hits'][j + rank_start]['score'] = cut_range[j]['score']
    return item


def permutation_pipeline(item=None, rank_start=0, rank_end=100, model_name=GPT_MODEL):
    messages = create_permutation_instruction(item=item, rank_start=rank_start, rank_end=rank_end)  # chan
    permutation = run_llm(messages)
    response = receive_permutation(item, permutation, rank_start=rank_start, rank_end=rank_end)
    return response


def sliding_windows(item=None, rank_start=0, rank_end=100, window_size=20, step=10, model_name='gpt-3.5-turbo',
                    api_key=None):
    item = copy.deepcopy(item)
    end_pos = rank_end
    start_pos = rank_end - window_size
    while start_pos >= rank_start:
        start_pos = max(start_pos, rank_start)
        item = permutation_pipeline(item, start_pos, end_pos, model_name=model_name)
        end_pos = end_pos - step
        start_pos = start_pos - step
    return item


def write_eval_file(rank_results, file):
    with open(file, 'w') as f:
        for i in range(len(rank_results)):
            rank = 1
            hits = rank_results[i]['hits']
            for hit in hits:
                f.write(f"{hit['qid']} Q0 {hit['docid']} {rank} {hit['score']} rank\n")
                rank += 1
    return True

# file_path = 'gpt/news.json'
# file_path = 'gpt/dl19.json'
# file_path = 'gpt/covid.json'
file_path = 'gpt/touche.json'
with open(file_path, 'r') as f:
    rank_results = json.load(f)
rank_results = rank_results

def evalute_dict(rank_dict:Dict[str,List[str]], the_topic:str): 
    """
    evaluate the rank_dict, one example is 
    rank_dict = {264014: ['4834547', '6641238', '96855', '3338827', '96851']}
    """
    # Evaluate nDCG@10
    temp_file = tempfile.NamedTemporaryFile(delete=False).name
    with open(temp_file, 'w') as f:
        for query_id,doc_ids_list in rank_dict.items():
            rank = 1    
            for doc_id in doc_ids_list:
                f.write(f"{query_id} Q0 {doc_id} {rank} {15-rank*0.1} rank\n")
                rank += 1
    return EvalFunction.eval(['-c', '-m', 'ndcg_cut.10', the_topic, temp_file])

# data = 'news'
# data = 'dl19'
# data = 'covid'
data = 'touche'
topic = THE_TOPICS[data]
rank_dict = {}
for line in rank_results:
    qid = line['qid']
    hits = line['hits']
    rank_dict[qid] = [hit['docid'] for hit in hits]
evalute_dict(rank_dict, topic)

new_results = []
for item in tqdm(rank_results):
    # new_item = permutation_pipeline(item, rank_start=0, rank_end=20, model_name=GPT_MODEL)
    new_item = sliding_windows(item, model_name=GPT_MODEL)
    new_results.append(new_item)

rank_dict = {}
for line in new_results:
    qid = line['qid']
    hits = line['hits']
    rank_dict[qid] = [hit['docid'] for hit in hits]
evalute_dict(rank_dict, topic)
breakpoint()
