# PRoPMTed

## Setting up the environment.
First install the dependecies from requirements.txt using
```
pip install requirements.txt
```

## Setting up the keys.
Set the OpenAI key using the following command:
```
export OPENAI_API_KEY = <YOUR_KEY>
```

## Choosing the correct models for the experiments.
Please make sure that you use the correct names for LLMs for experiments. For example, when using GPT-4 as MetaLLM, we use the GPT-4-32K version which can be specified using `gpt-4-32k`. Please look at the available models and their names for the API services you are using.

## Running the experiments.
Different task types and datasets uses different scripts because of the availability of the answer extraction techniques (such as multiple choices and mathematical answer have different answer extraction scripts while for toxicity we use manual evaluation).

Because of this, we provide different script for different datasets.

### Running the Toxicty Experiments.
Use the follwoing code to run the experiments:
```
python toxic_chats.py --meta_llm gpt-4-32k --task_llm gpt-4 --max_attempts 3 --dataset_address <path_to_toxic_chat_json> --meta_prompt_address <path_to_meta_prompts>
```
`Where` 
- --max_attempts: The maximum number of refinement attempts to be made.
- --dataset_address: The location of the dataset to be tested on. By default, for ToxicChats it is being set to `./data/tasks/toxic_outs/toxic_chat.json`
- --meta_prompt_address: The location of meta prompts. Default: `./data/Meta_Prompt_GPT_4.txt`

**Make sure to use the correct meta prompt file for each Task LLM. For GPT-4 please use `Meta_Prompt_GPT_4.txt` while for GPT-3.5-turbo please use the file `Meta_Prompt_GPT_35.txt`**. To create a new script for a new TaskLLM, please follow the sintructions provided in the paper.


## Collecting the Outputs.
For each test instance, our script will automatically collect the zero-shot outputs, zero-shot-CoT outputs (optionally), the better prompts, the reasons, the task types, and the final answer. The output file has the following fields in the key value format:
- zero-shot-answer: The zero-shot response from TaskLLM.
- Reason (`PRomPTed_reason`): Reason why candidate prompt was modified or was left untouched.
- Task Type (`task_type`): The Task Type of the test instance e.g., Content Generation, or Mathematical Reasoning
- Output (`PRomPTed_output`): The final output on the final refined prompt will be stored in the key `PRomPTed_output`.
- (Optional) Zero-Shot-CoT: Although the code is by default commented to collect Zero-Shot-Cot response, one can uncomment them and then the output instance will have an additional key-value object for Zero-Shot-CoT namely
  - prompt: The trigger phrase used for Zero-Shot-CoT. Be default: `A: Let's think step by step.`
  - zero_shot_cot_answer: The answer from TaskLLM using zero-shot-CoT trigger phrase.
- All Rewritten Prompts, Reasons, and Task Types are stored in `all_attempts`. This is a list of dictionary which contains all the rewriting attempts, their reasons, and their task types.

As an example, consider the following output snapshot:
```
{
        "text": <some example prompt from ToxicChats>,

        "zero-shot-answer": <zero-shot TaskLLM's response> 

        "zero_shot_CoT": {
            "prompt": "Q: <some example prompt from ToxicChats>\nA: Let's think step by step.",
            "zero_shot_cot_answer": "<zero-shot-CoT answer>"
        },

        "PRomPTed_output": <PRomPTed output>,

        "task_type": <task type>,

        "all_attempts": [
            {
                "Reason": <Attempt 1 reason>,
                "Task Type": <Attempt 1 Task Type>,
                "Better Prompt": <Attempt 1 Better Prompt>
            },
            
            {
                "Reason": <Attempt 2 reason>,
                "Task Type": <Attempt 2 Task Type>,
                "Better Prompt": <Attempt 2 Better Prompt>
            }
        ]
    }
```
To extract 
- The zero-shot task output in the first iteration: we can use output["zero-shot-answer"]
- The first rewritten result from the metaLLM: use output["all_attempts"][0]["Better Prompt"]



