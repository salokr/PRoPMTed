import json, os
import backoff, requests
import openai
from tqdm import tqdm
import tiktoken
import argparse
from openai import OpenAI, AzureOpenAI
client = OpenAI()

encoder=tiktoken.encoding_for_model('gpt-4')

parser = argparse.ArgumentParser(description="arguments for PRomPTed")
parser.add_argument("--meta_llm", type = str, default = "gpt-4-32k", help = "The Meta LLM for rewriting prompts")
parser.add_argument("--task_llm", type = str, default = "gpt-4", help = "The Task LLM for rewriting prompts")
parser.add_argument("--max_attempts", type = int, default = "3", help = "Max number of iterations to run")
parser.add_argument("--dataset_address", type = str, default = "./data/tasks/toxic_outs/toxic_chat.json", help = "Dataset to run on")
parser.add_argument("--meta_prompt_address", type = str, default = "./data/Meta_Prompt_GPT_4.txt", help = "Meta prompt address")
parser.add_argument("--api_type", type=str,default="openai", help = "azure or openai APIs are only supported")
args = parser.parse_args()

if(args.api_type=="azure"):
    client = AzureOpenAI()
    openai.api_type = "[PLEASE SET]"
    openai.api_base = "[PLEASE SET]"
    openai.api_version = "[PLEASE SET]"

#if not azure, then we'll use OpenAI's service as default
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
        response = client.chat.completions.create(model=model_name,messages=[{"role": "user","content": prompt}], temperature=.7,  top_p=.7,frequency_penalty=0,presence_penalty=0, stop=None)
        # print(response)
        response = response.choices[0].message.content
        if(count_tokens):
            rewrite_token_count += len(encoder.encode(prompt))
    except Exception as e:
        print("Exception", e)
    return response

def stopping_criterion(reason):
    if("is correct" in reason.lower() or reason.lower().strip().startswith("correct") or reason.lower().strip().startswith("well-crafted")or reason.lower().strip().startswith("well-alligned")or reason.lower().strip().startswith("clear")or reason.lower().strip().startswith("a well-crafted")):
        return True
    return False

def extract_answer(response):
    # Please sepcify the answer extraction technique/script here. You can either use the hard matching like "The Answer is .*." regex or the zero-shot-cot technique.
    return response



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


dataset_name = os.path.basename(dataset_address)
dataset_name = dataset_name[:dataset_name.find(".")]
def main():
    global rewrite_calls, rewrite_token_count, dataset_name
    jsn = json.load(open(json_address))
    meta_prompt = ''.join([lines for lines in open(meta_prompt_address)]) + "\n\n### Candidate Prompt ###\n"
    for _, j in tqdm(enumerate(jsn), total = len(jsn), desc = f"Processing {dataset_name}"):
        question = j["Question"] # Please change the field name to appropriate prompt field based on the dataset. For example, for BigBench, tasks you may want to use question = jsn["examples"][idx]["Question"]
        input_prompt = "" # This is used for any additional task instrutions you may want to specify. For example, you can specify that the task expects an answer in Multi-Choice questions format.
        question = (input_prompt + question.strip()).strip() # Then the prompt will become the additional task instructions plus the question.
        # gt_answer = j["label"] #if we want to compare the groud truth with the prompted answer then we can specify the GT field here and later we can use it for comparison.
        answer_found = False #This flag is used to break the loop when the MetaLLMs is satisfied with the answer and the prompt.
        attempts = []#used to store all the attempts.
        print(question)
        i, max_taskLLM_tries = 0, 0
        while i < MAX_ATTEMPTS:
            try:
                task_llm_response = gpt_response(question, task_llm) 
            except:
                max_taskLLM_tries += 1
                if(max_taskLLM_tries>=MAX_ATTEMPTS):
                    task_llm_response = "Can't generate a response."#TaskLLM is unable to generate any response to this prompt maybe because it is too toxic.
                else:
                    continue
            print("--------------------------------------------------------------------------")
            print(f"Question: {question}\n{task_llm} Response: {task_llm_response}")
            prompt = meta_prompt + question + f"\n\n### Output ###\n{task_llm_response}\n\n### Reason ###\nThe output is "
            print("--------------------------------------------------------------------------")
            rewrite_calls += 1
            if(i==0):#for the first call to the LLMs, we store that as the zero-shot output.
                # print(prompt)
                print("[PROMPT HIDDEN FOR CLARITY]")
                rewrite_token_count += len(encoder.encode(prompt))
                j["zero-shot-answer"] = task_llm_response
            else:
                print(f"----------------Attempt: {i+1}---------------")
            try:
                reason, task_type, better_prompt = get_full_responses(prompt, rewrite_llm)
            except Exception as e2:
                print("Rewrite exception, attempting Again. Error Type: ", e2)
                continue
            i += 1
            attempts.append({"Reason": reason, "Task Type": task_type, "Better Prompt": better_prompt, "Zero-Shot-Response": task_llm_response})
            print("--------Reason--------")
            print(reason)
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
        with open(f"./outputs/{dataset_name}_task_LLM_{task_llm}_meta_LLM_{rewrite_llm}.json", "w") as f:
            json.dump(jsn, f, indent = 4)
        break

    print(f"Avg. number of tokens: {rewrite_token_count/len(jsn)}\nTotal Rewrite Calls: {rewrite_calls}")
    """
    For bookkeeping
    -------------------------
    Avg. number of tokens: 
    Total Rewrite Calls: 
    """



if __name__ == '__main__':
    main()
