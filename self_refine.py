import openai
MAX_ATTEMPT = 5
task_llm = refine_llm = "gpt-3.5-turbo"
feedback_llm = "gpt-4"
# openai.api_key = "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"
openai.api_key = "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"
def get_response(prompt, model):
    print("="*100, 'START', "="*100)
    print("-------PROMPT-------")
    print(prompt)
    response = openai.ChatCompletion.create(model=model,messages=[{"role": "user","content": prompt}], temperature=0.7,  top_p=1,frequency_penalty=0,presence_penalty=0)
    response = response["choices"][0]["message"]["content"]
    print("------RESPONSE------")
    print(response)
    print("="*200)
    print("="*100, 'END', "="*100)
    print(f"Utilized model: {model}")
    _ = input("Model correct?")
    return response

"""
def self_refine(prompt: str) -> str:
    def is_refinement_sufficient(prompt, feedback, initial, refined) -> bool:
        # Define stopping criteria here
        pass

    answer = GPT(prompt)

    while True:
        feedback = GPT(feedback_prompt, answer)
        if is_refinement_sufficient(prompt, feedback, answer, refined):
            break
        refined = GPT(refiner_prompt, feedback, answer)
        answer = refined

    return refined
"""
def self_refine(prompt, stopping_criteria, feedback_prompt, refiner_prompt):
    def is_refinement_sufficient(prompt, feedback, refined, step) -> bool:
        return stopping_criteria(prompt, feedback, refined, step)
    answer = get_response(prompt, task_llm)# "### Attempt 1 ###\n" + get_response(prompt, task_llm)
    attempt = 1
    full_prompt = prompt.strip() + "\n" + answer.strip() + "\n"
    while True:
        feedback = get_response(feedback_prompt.strip() + "\n\n" + full_prompt, feedback_llm) #f"### Feedback {attempt} ###\n" + get_response(feedback_prompt.strip() + "\n\n" + full_prompt, feedback_llm)#f_bt = M(p_fb||x||y_t)
        if is_refinement_sufficient(prompt, feedback, answer, attempt):
            break
        attempt += 1
        full_prompt += feedback.strip() + "\n" 
        refine_prompt = refiner_prompt.strip() + "\n" + full_prompt #refiner_prompt.strip() + "\n" + full_prompt + f"### Attempt {attempt} ###\n"
        answer = get_response(refine_prompt, task_llm)
        full_prompt += refine_prompt + answer
    return answer

question = "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?"

def stopping_criteria(prompt, feedback, refined, step):
    # feedback = feedback.split("###")[-1]
    # print(">>", feedback, feedback.find("### STOP ###"), [step>=MAX_ATTEMPT, [(feedback.find(flag)>=0) for flag in solution_found_flags]])
    # print("--------")
    if(any([step>=MAX_ATTEMPT, any([feedback.find(flag)>=0 for flag in solution_found_flags])])):
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> step <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<", step)
        return True
    return False

task_instructions = ""

feedback_prompt = ''.join([lines for lines in open("data/self_refine_adapted_prompts.txt")])
refiner_prompt = "Okay! Here is the rewrite:"
solution_found_flags = ["### STOP ###", "### END ###"]
prompt = question#task_instructions.replace("###QUESTION_HERE###", question)
self_refine(prompt, stopping_criteria, feedback_prompt, refiner_prompt)