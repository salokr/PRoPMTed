import json
from prompts import prompt
import backoff, requests


meta_prompt_address = "/Volumes/Academic/Projects/PRoPMTed/data/Meta_Prompt_GPT_4.txt"
json_address = "/Users/saurabhsrivastava/Downloads/self-refine-main/data/tasks/gsm/gsm.jsonl"
MAX_ATTEMPTS = 3
rewrite_llm = task_llm = "gpt-4"


def gpt_response(prompt, model_name, text = None):
	if(text is not None):
		return text
	return """The candidate prompt for disease name extraction has several limitations and the better prompt should handle them in the following ways:
1) Specificity in Disease Name Handling: The bad prompt had vague instructions for handling compound disease names and repetitive mentions, leading to inaccuracies. In contrast, the refined prompt should have clear guidelines for these issues, ensuring accurate treatment of complex diseases.
2) Inclusion and Exclusion Criteria: The bad prompt omitted important exclusions like genetic, biological terms terms and descriptive modifiers, resulting in irrelevant or overly detailed extractions. The new prompt should explicitly state these exclusions, focusing solely on relevant disease names and descriptors.
3) Contextual Accuracy in Disease Identification: The bad prompt failed to address the context of disease mentions, leading to misunderstandings. The improved prompt should emphasize the contextual presentation of disease names to ensure accurate and contextually appropriate extractions.
4) Precision in Disease Name Extraction: The bad prompt lacked guidance on precise extractions, causing over- or under-extractions. The refined prompt should define what constitutes a disease name and key medical descriptor clearly, avoiding the inclusion of irrelevant information.

### Task Type ###
[Information Extraction] [Domain-Specific Task]

### Better Prompt ###
Your task is to extract the names of diseases or medical conditions and their key medical descriptors from the provided sentence. Adhere to these guidelines:
1) Extract Disease Names and Specific Medical Descriptors: Focus on extracting both disease names and medical descriptors that specifically define the type or nature of the disease, such as 'autosomal codominant disorder'.
2) Limit Extraction to Disease Characterization: Extract only those terms that directly contribute to characterizing the disease. Avoid extending the extraction to include symptoms, outcomes, or biochemical details unless they are explicitly part of the disease's name or classification.
3) Exclude Genetic and Biological Terms Not Integral to Disease Names: Do not extract terms related to genes, proteins, or genetic markers unless they form an integral part of the disease's name or classification.
4) Precise Contextual Extraction: Ensure that the extracted information is a precise representation of how the diseases and their key descriptors are mentioned in the sentence, avoiding any additions or omissions.
Apply these refined guidelines to the following sentence to extract the disease names and their specific medical descriptors:
\"Familial hypobetalipoproteinemia is an autosomal codominant disorder resulting in a dramatic reduction in plasma concentrations of apolipoprotein (apo) B, cholesterol, and beta-migrating lipoproteins.\"
Provide your answer in the following format: \"The answer is [YOUR_ANSWERS]\""""

def stopping_criterion(reason):
	if("is correct" in reason.lower()):
		return True
	return False

def extract_answer(response):
	return "final response"

test_prompt = prompt("candidate_prompt", "reason", "prompt_type", "output", "better_prompt")

jsn = [json.loads(x) for x in open(json_address)]
meta_prompt = ''.join([lines for lines in open(meta_prompt_address)]) + "\n\n### Candidate Prompt ###\n"

#OpenAI method to retry on failed prompt calls
@backoff.on_exception(backoff.expo, Exception, max_tries=3, max_time = 61)
def get_full_responses(prompt, task_llm):
	response = gpt_response(prompt, rewrite_llm)
	response_parts = response.split("###")
	reason, task_type, better_prompt = response_parts[0].strip(), response_parts[2].strip(), response_parts[4].strip()
	return reason, task_type, better_prompt


for j in jsn:
	question = j["input"]
	gt_answer = j["target"]
	answer_found = False
	for i in range(MAX_ATTEMPTS):
		task_llm_response = gpt_response(question, task_llm, "new_question")
		prompt = meta_prompt + question + f"\n\n### Output ###\n{task_llm_response}\n\n### Reason ###\nThe output is "
		print(prompt)
		# _ = input("Continue?")
		# response = gpt_response(prompt, rewrite_llm)
		# response_parts = response.split("###")
		reason, task_type, better_prompt = get_full_responses(prompt, rewrite_llm)
		# print("reason###", reason, "###")
		# print("task_type###", task_type, "###")
		# print("better_prompt###", better_prompt, "###")
		if(stopping_criterion(reason)):
			answer_found = True
			answer = task_llm_response
			break
		question = better_prompt
	if(not answer_found):
		answer = gpt_response(question, task_llm, "new_answer")
	final_answer = extract_answer(answer)
	j["PRomPTed_output"] = final_answer
	print(final_answer)
	break




