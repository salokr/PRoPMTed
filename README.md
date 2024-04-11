# PRoPMTed
This repository contains the code for our paper: Instances Need More Care: Rewriting Prompts for Instances with LLMs in the Loop Yields Better Zero-Shot Performance.

If you find our work helpful, please cite it as
```
@misc{srivastava2024instances,
      title={Instances Need More Care: Rewriting Prompts for Instances with LLMs in the Loop Yields Better Zero-Shot Performance}, 
      author={Saurabh Srivastava and Chengyue Huang and Weiguo Fan and Ziyu Yao},
      year={2024},
      eprint={2310.02107},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
```
## Setting up the environment.
This project is tested in Python 3.9.6.

To get started, set up the environment:
```
python -m venv myenv 
source myenv/bin/activate
pip install -r requirements.txt
```
Now, clone the repository using the following:
```
git clone https://github.com/salokr/PRoPMTed.git
cd PRoPMTed
mkdir outputs
```
## Setting up the keys.
Set the OpenAI key using the following command:
```
export OPENAI_API_KEY = <YOUR_KEY>
```
 Note*: Please note that our code supports only OpenAI and Azure API calls. If you intend to use Azure API, please additionally specify the api_version, api_base, and api_version in the code.

## Running the experiments.
Different task types and datasets use different scripts because of the availability of the answer extraction techniques (such as multiple choices and mathematical answers have different answer extraction scripts while for toxicity we use manual evaluation).

Because of this, we provide different scripts for different datasets.

### Running the Toxicity Experiments.
Use the following code to run the experiments:
```
python toxic_chats.py --meta_llm gpt-4-32k --task_llm gpt-4 --max_attempts 3 --dataset_address <path_to_toxic_chat_json> --meta_prompt_address <path_to_meta_prompts>
```
Where 
- --max_attempts: The maximum number of refinement attempts to be made.
- --dataset_address: The location of the dataset to be tested on. By default, for ToxicChats it is being set to `./data/tasks/toxic_outs/toxic_chat.json`
- --meta_prompt_address: The location of meta prompts. Default: `./data/Meta_Prompt_GPT_4.txt`

**Make sure to use the correct meta prompt file for each Task LLM. For GPT-4 please use `Meta_Prompt_GPT_4.txt` while for GPT-3.5-turbo please use the file `Meta_Prompt_GPT_35.txt`**. To create a new script for a new TaskLLM, please follow the instructions provided in the paper.

### In the following table, we enumerate parameters to run our code for ToxicChats.
|    **Meta LLM**    |    **Task LLM**    |    **Meta Prompt Address**    |                                                                       **Command**                                                                      |
|:------------------:|:------------------:|:-----------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------:|
|      gpt-4-32k     |        gpt-4       |  ./data/Meta_Prompt_GPT_4.txt |             python toxic_chats.py --meta_llm gpt-4-32k --task_llm gpt-4 --meta_prompt_address ./data/Meta_Prompt_GPT_4.txt --max_attempts 3            |
| gpt-3.5-turbo-1106 |        gpt-4       |  ./data/Meta_Prompt_GPT_4.txt |        python toxic_chats.py --meta_llm gpt-3.5-turbo-1106 --task_llm gpt-4 --meta_prompt_address ./data/Meta_Prompt_GPT_4.txt --max_attempts 3        |
| gpt-3.5-turbo-1106 | gpt-3.5-turbo-1106 | ./data/Meta_Prompt_GPT_35_Turbo.tx | python toxic_chats.py --meta_llm gpt-3.5-turbo-1106 --task_llm gpt-3.5-turbo-1106 --meta_prompt_address ./data/Meta_Prompt_GPT_35_Turbo.tx --max_attempts 3 |

## Collecting the Outputs.
Our code saves the outputs in a file with the naming convention as `./outputs/<task_name>_task_LLM_<TaskLLM>_meta_LLM_<MetaLLM>.json`.
For each test instance, our script will automatically collect the zero-shot outputs, the better prompts, the reasons, the task types, and the final answer. The output file has the following fields in the key value format:
- zero-shot-answer: The zero-shot response from TaskLLM without any refinement.
- Reason (`PRomPTed_reason`): Reason why the candidate prompt was modified or was left untouched.
- Task Type (`task_type`): The Task Type of the test instance e.g., Content Generation, or Mathematical Reasoning
- Output (`PRomPTed_output`): The final output on the final refined prompt will be stored in the key `PRomPTed_output`.
- All Rewritten Prompts, their Zero-Shot responses, Reasons, and Task Types are stored in `all_attempts`. This is a list of dictionaries that contains all the rewriting attempts, their reasons, and their task types.

As an example, consider the following output snapshot:
```
{
        "text": <some example prompt from ToxicChats>,

        "zero-shot-answer": <zero-shot TaskLLM's response>, 

        "PRomPTed_output": <PRomPTed output>,

        "task_type": <task type>,

        "all_attempts": [
            {
                "Reason": <Attempt 1 reason>,
                "Task Type": <Attempt 1 Task Type>,
                "Better Prompt": <Attempt 1 Better Prompt>,
                "Zero-Shot-Response": <Attempt 1 zero-shot response>
            },
            
            {
                "Reason": <Attempt 2 reason>,
                "Task Type": <Attempt 2 Task Type>,
                "Better Prompt": <Attempt 2 Better Prompt>,
                "Zero-Shot-Response": <Attempt 2 zero-shot response>
            }
        ]
    }
```
If you want to extract different fields of output at index `idx`. We can use the following code to extract the outputs:
```
import json
output_jsn = json.load(open(<output_file_address>))
for idx in range(len(output_jsn)):
    # Extract the first-iteration Zero-Shot (Response from TaskLLM without refinement)
    print(output_jsn[idx]["zero-shot-answer"])
    # The first rewritten prompt from the MetaLLM
    print(output_jsn[idx]["all_attempts"][0]["Better Prompt"])
    print("-"*100)
```
 
## Creating a custom data loader and custom PRomPTed script.
In addition, we also provide a `PRomPTed.py` which can be used to run on any other dataset of your choice. To run PRomPTed on your dataset, please make the following changes to PRomPTed.py:
The code assumes that the input is a JSON file with the following format:
```
[
     {"Question": <prompt>, "Other fields such as Ground Truth Outputs, etc.": <fields values>},
     {"Question": <prompt>, "Other fields such as Ground Truth Outputs, etc.": <fields values>},
     ...
]
``` 
If you need to use any other data loader, you can use the following scripts:
1) For JSONL formats use the following:
```
jsn = [json.loads(x) for x in open(dataset_address)]
```
2) For BigBench tasks:
```
jsn = json.load(open(dataset_address))["examples"]
```
3) Any other:
Please create your custom data loader function and specify the test-instance field at line 88 in PRomPTed.py
