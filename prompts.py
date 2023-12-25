class prompt:
    def reorder_objects(self, style, mapping):
        ordered_vars = [mapping[char] for char in style]
        return "\n".join(ordered_vars)
    def __init__(self, candidate_prompt, reason, prompt_type, output, better_prompt, style="corpb"):
        self.candidate_prompt = candidate_prompt
        self.reason = reason
        self.prompt_type = prompt_type
        self.output = output
        self.better_prompt = better_prompt
        self.style = style
        self.mapping = {"c":self.candidate_prompt, "r":self.reason, "p":self.prompt_type, "o":self.output, "b":self.better_prompt}
        self.prompt = self.reorder_objects(self.style, self.mapping)
    def extract_section(self, section_name, text = None):
        text = self.prompt if text == None else text
        # Define the start marker for the section
        start_marker = f"###{section_name}###"
        # Find the start of the section
        start_index = text.find(start_marker)
        if start_index == -1:
            return "Section not found"
        # Adjust the start index to get the text after the marker
        start_index += len(start_marker)
        # Try to find the next section marker, indicating the end of the current section
        end_index = text.find("###", start_index)
        # If there's no next section marker, the end is the end of the string
        if end_index == -1:
            section_content = text[start_index:].strip()
        else:
            # Extract the section content
            section_content = text[start_index:end_index].strip()
        return section_content


test_prompt = prompt("candidate_prompt", "reason", "prompt_type", "output", "better_prompt")

def get_batched_prompts(meta_prompt_address = "new_prompts.txt"):
	f = open(meta_prompt_address, "r")
	meta_prompt_full = ''.join([lines for lines in f])
	meta_prompts = meta_prompt_full.split("\n\n")
	task_instructions = meta_prompts[0]
	all_candidate_prompts, all_reasons, all_prompt_types, all_outputs, all_better_prompts = "", "", "", "", ""
	batch_prompts = task_instructions.strip() + "\n\n" 
	for idx, meta_prompt in enumerate(meta_prompts[1:]):
		candidate_prompt, reason, prompt_type, output, better_prompt = test_prompt.extract_section("Candidate Prompt", meta_prompt), test_prompt.extract_section("Reason", meta_prompt), test_prompt.extract_section("Better Prompt Type", meta_prompt), test_prompt.extract_section("Output", meta_prompt), test_prompt.extract_section("Better Prompt", meta_prompt)
		all_candidate_prompts += f"Candidate Prompt {idx+1}: " + candidate_prompt.strip() + "\n" 
		all_reasons += f"Reason {idx+1}: " + reason.strip() + "\n"
		all_prompt_types += f"Task Type {idx+1}: " + prompt_type.strip() + "\n"
		all_outputs += f"Output {idx+1}: " + output.strip() + "\n"
		all_better_prompts += f"Better Prompt {idx+1}: " + better_prompt.strip() + "\n"
	batch_prompts += "\n\n".join([all_candidate_prompts, all_outputs, all_reasons, all_prompt_types, all_better_prompts])
	return batch_prompts





