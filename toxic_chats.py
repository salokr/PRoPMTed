import json, os
import backoff, requests
import openai
from tqdm import tqdm
import tiktoken

import argparse
encoder=tiktoken.encoding_for_model('gpt-4')

parser = argparse.ArgumentParser(description="arguments for PRomPTed")
parser.add_argument("--meta_llm", type = str, default = "gpt-4", help = "The Meta LLM for rewriting prompts")
parser.add_argument("--task_llm", type = str, default = "gpt-4", help = "The Task LLM for rewriting prompts")
parser.add_argument("--max_attempts", type = int, default = "3", help = "Max number of iterations to run")
parser.add_argument("--dataset_address", type = str, default = "./data/tasks/toxic_outs/toxic_chat.json", help = "Dataset to run on")
parser.add_argument("--meta_prompt_address", type = str, default = "./data/Meta_Prompt_GPT_4.txt", help = "Meta prompt address")

args = parser.parse_args()

openai.api_type = "azure"
openai.api_base = "https://openai-access-canada-east.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key = os.getenv("OPENAI_API_KEY")


meta_prompt_address = args.meta_prompt_address
json_address = args.dataset_address
MAX_ATTEMPTS = args.max_attempts
rewrite_llm  = args.meta_llm
task_llm = args.task_llm


rewrite_token_count = 0
rewrite_calls = 0
#OpenAI method to retry on failed prompt calls
@backoff.on_exception(backoff.expo, Exception, max_tries=3, max_time = 63)
def gpt_response(prompt, model_name, count_tokens = True, text = None):
    global rewrite_token_count
    response = ""
    try:
        response = openai.ChatCompletion.create(engine=model_name,messages=[{"role": "user","content": prompt}], temperature=.7,  top_p=.7,frequency_penalty=0,presence_penalty=0, stop=None)
        response = response["choices"][0]["message"]["content"]
        if(count_tokens):
            rewrite_token_count += len(encoder.encode(prompt))
    except Exception as e:
        print("Exception", e)
    return response

def stopping_criterion(reason):
    if("is correct" in reason.lower() or reason.lower().strip().startswith("correct")):
        return True
    return False

def extract_answer(response):
    # Note 1: For toxic chats, we don't have an evaluation metrics. 
    # Note 2: Since this is a content generation task, We don't have a format specifier
    # Therefore, we return the output as is and the evaluation is done manually.
    # But, we can specify any other library here which can extract the answer or evaluate the toxicity here.
    return response


jsn = json.load(open(json_address))
meta_prompt = ''.join([lines for lines in open(meta_prompt_address)]) + "\n\n### Candidate Prompt ###\n"

#OpenAI method to retry on failed prompt calls
@backoff.on_exception(backoff.expo, Exception, max_tries=3, max_time = 63)
def get_full_responses(prompt, llm):
    try:
        response = gpt_response(prompt, llm)
        response_parts = response.split("###")
        reason, task_type, better_prompt = response_parts[0].strip(), response_parts[2].strip(), response_parts[4].strip()
    except Exception as e:
        print("Error: ", e)
        reason, task_type, better_prompt = "", "", ""
        raise Exception("Exception e:", e)
    return reason, task_type, better_prompt


for _, j in tqdm(enumerate(jsn), total = len(jsn), desc = "Processing ToxicChats"):
    question = j["text"]
    input_prompt = ""#any task instruction goes here
    question = (input_prompt + question.strip()).strip()
    # gt_answer = j["label"]
    answer_found = False
    attempts = []
    print(question)
    for i in range(MAX_ATTEMPTS):
        try:
            task_llm_response = gpt_response(question, task_llm) 
        except:
            task_llm_response = "Can't generate a response."
        print("--------------------------------------------------------------------------")
        print(f"Question: {question}\n{task_llm} Response: {task_llm_response}")
        prompt = meta_prompt + question + f"\n\n### Output ###\n{task_llm_response}\n\n### Reason ###\nThe output is "
        print("--------------------------------------------------------------------------")
        rewrite_calls += 1
        if(i==0):
            print(prompt)
            print("[PROMPT HIDDEN FOR CLARITY]")
            rewrite_token_count += len(encoder.encode(prompt))
            j["zero-shot-answer"] = task_llm_response
            # zero_shot_cot_response = {"prompt": f"Q: {question}\nA: Let's think step by step.", "zero_shot_cot_answer": ""}
            # try:
            #     print("=-"*50)
            #     print(f"Zero-Shot-CoT-Question: {zero_shot_cot_response['prompt']}")
            #     zero_shot_cot_response["zero_shot_cot_answer"] = gpt_response(zero_shot_cot_response["prompt"], task_llm, count_tokens = False)
            #     print(f"Zero-Shot-CoT Response: {zero_shot_cot_response['zero_shot_cot_answer']}")
            #     print("=-"*50)
            # except Exception as e:
            #     print(f"Zero-Shot-CoT error: {e}")
            #     pass
            j["zero_shot_CoT"] = zero_shot_cot_response
            print("=-"*50)
        else:
            print(f"----------------Attempt: {i+1}---------------")
        try:
            reason, task_type, better_prompt = get_full_responses(prompt, rewrite_llm)
        except Exception as e2:
            print("E2 exception: ", e2)
            reason, task_type, better_prompt = "", "", question
        attempts.append({"Reason": reason, "Task Type": task_type, "Better Prompt": better_prompt})
        print("--------Reason--------")
        print(reason)
        # print("task_type###", task_type, "###")
        print("--------Better Prompt--------")
        print(better_prompt) 
        print("--------------------------------------------------------------------------")
        if(stopping_criterion(reason)):
            answer_found = True
            answer = task_llm_response
            break
        question = better_prompt
    if(not answer_found):
        answer = gpt_response(question, task_llm)
    final_answer = extract_answer(answer)
    j["PRomPTed_output"] = final_answer
    j["PRomPTed_reason"] = reason
    j["task_type"] = task_type
    j["all_attempts"] = attempts
    print("--------Predicted Answer--------")
    print(final_answer)
    print("--------Predicted Answer--------")
    with open(f"./outputs/toxic_chat_task_LLM_{task_llm}__meta_LLM_{rewrite_llm}.json", "w") as f:
        json.dump(jsn, f, indent = 4)



print(f"Avg. number of tokens: {rewrite_token_count/len(jsn)}\nTotal Rewrite Calls: {rewrite_calls}")
"""
Avg. number of tokens: 
Total Rewrite Calls: 


"""
