import json, os
import backoff, requests
import openai
from tqdm import tqdm
openai.api_type = "azure"
openai.api_base = "https://openai-access-canada-east.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key = os.getenv("OPENAI_API_KEY")


meta_prompt_address = "./data/Meta_Prompt_GPT_4.txt"
json_address = "./data/tasks/gsm/gsm.jsonl"
MAX_ATTEMPTS = 3
rewrite_llm  = "gpt-4-32k"
task_llm = "gpt-4"

def gpt_response(prompt, model_name, text = None):
    response = openai.ChatCompletion.create(engine=model_name,messages=[{"role": "user","content": prompt}], temperature=.7,  top_p=.7,frequency_penalty=0,presence_penalty=0, stop=None)
    response = response["choices"][0]["message"]["content"]
    return response

def stopping_criterion(reason):
	if("is correct" in reason.lower()):
		return True
	return False

def extract_answer(response):
	return response


jsn = [json.loads(x) for x in open(json_address)]#[:10]
meta_prompt = ''.join([lines for lines in open(meta_prompt_address)]) + "\n\n### Candidate Prompt ###\n"

#OpenAI method to retry on failed prompt calls
@backoff.on_exception(backoff.expo, Exception, max_tries=3, max_time = 63)
def get_full_responses(prompt, task_llm):
	try:
		response = gpt_response(prompt, rewrite_llm)
	except Exception as e:
		print("Error: ", e)
	response_parts = response.split("###")
	reason, task_type, better_prompt = response_parts[0].strip(), response_parts[2].strip(), response_parts[4].strip()
	return reason, task_type, better_prompt


for _, j in tqdm(enumerate(jsn), total = len(jsn), desc = "Processing GSM-8K"):
	question = j["input"]
	gt_answer = j["target"]
	answer_found = False
	attempts = []
	for i in range(MAX_ATTEMPTS):
		task_llm_response = gpt_response(question, task_llm)
		print("--------------------------------------------------------------------------")
		print(f"Question: {question}\n{task_llm} Response: {task_llm_response}")
		prompt = meta_prompt + question + f"\n\n### Output ###\n{task_llm_response}\n\n### Reason ###\nThe output is "
		print("--------------------------------------------------------------------------")
		if(i==0):
			print(prompt)
		else:
			print(f"----------------Attempt: {i+1}---------------")
		# _ = input("Continue?")
		# response = gpt_response(prompt, rewrite_llm)
		# response_parts = response.split("###")
		try:
			reason, task_type, better_prompt = get_full_responses(prompt, rewrite_llm)
		except:
			reason, task_type, better_prompt = "", "", question
		attempts.append({"Reason": reason, "Task Type": task_type, "Better Prompt": better_prompt, "Prompt": prompt})
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
	print("--------Answer--------")
	print(final_answer)
	print("--------Answer--------")
	# break

with open("./outputs/gsm8k_prompted.jsonl", "w") as f:
	for j in jsn:
		f.write(json.dumps(j) + "\n")

try:
	jsn = full_outs
except:
	pass


for j in jsn:
	print("Question: ", j["input"])
	print("="*100)
	print("Attempts: 1", j["all_attempts"][0]["Better Prompt"])
	print("="*100)
	try:
		print("Attempts: 2", j["all_attempts"][1]["Better Prompt"])
		print("="*100)
	except:
		pass
	try:
		print("Attempts: 3", j["all_attempts"][2]["Better Prompt"])
		print("="*100)
	except:
		pass
	print("Predicted Output")
	print(j["PRomPTed_output"])
	print("-"*100)
	print("Ground Truth Output")
	print(j["target"])
	print("-"*100)
	_ = input("Continue???")
	_ = os.system("clear")


