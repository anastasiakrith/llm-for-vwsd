import re

from modules.llms import GPT_3, GPT_3_5, BLOOMZ, OPT, VICUNA, LLAMA
from modules.captioners import BLIP, GIT, VIT_GPT2
from modules.prompts import no_CoT_prompt, think_prompt, CoT_prompt, choose_no_CoT_prompt, choose_CoT_prompt
from modules.few_shot_selection import FewShotSelectionModule

AVAILABLE_LLMS = {
    'gpt-3.5': lambda: GPT_3_5(),
    'vicuna-13b': lambda: VICUNA(nparameters='13B'),
}

AVAILABLE_CAPTIONERS = {
    'blip': lambda strategy: BLIP(strategy=strategy, large=False),
    'blip_large': lambda strategy: BLIP(strategy=strategy, large=True),
    'git': lambda strategy: GIT(strategy=strategy, large=False),
    'git_large': lambda strategy: GIT(strategy=strategy, large=True),
    'vit_gpt2': lambda strategy: VIT_GPT2(strategy=strategy)
}

AVAILABLE_ZERO_SHOT_QA_PROMPT_TEMPLATES = {
    'no_CoT': no_CoT_prompt,
    'CoT': think_prompt,
    'choose_no_CoT': choose_no_CoT_prompt,
    'choose_CoT': choose_CoT_prompt
}

AVAILABLE_FEW_SHOT_QA_PROMPT_TEMPLATES = {
    'no_CoT': no_CoT_prompt
}

class ZeroShotQAModule:

    def __init__(self, llm, captioner, strategy, prompt_template):

        if llm not in AVAILABLE_LLMS:
            raise ValueError(f"Invalid LLM: {llm}. Should be one of {AVAILABLE_LLMS.keys()}")
        
        if captioner not in AVAILABLE_CAPTIONERS:
            raise ValueError(f"Invalid LLM: {captioner}. Should be one of {AVAILABLE_CAPTIONERS.keys()}")

        if strategy not in ['greedy', 'beam']:
            raise ValueError(f"Invalid captioner strategy: {strategy}. Should be one of ['greedy', 'beam']")
        
        if prompt_template not in AVAILABLE_ZERO_SHOT_QA_PROMPT_TEMPLATES:
            raise ValueError(f"Invalid prompt template: {prompt_template}. Should be one of {AVAILABLE_ZERO_SHOT_QA_PROMPT_TEMPLATES.keys()}")

        self.strategy = strategy
        self.llm = AVAILABLE_LLMS[llm]()
        self.captioner = AVAILABLE_CAPTIONERS[captioner](strategy)
        self.CoT_flag = prompt_template == 'CoT' 
        self.prompt = AVAILABLE_ZERO_SHOT_QA_PROMPT_TEMPLATES[prompt_template]

    def parse_answer(self, answer):
        pred = re.findall(r'\(A\)|\(B\)|\(C\)|\(D\)|\(E\)|\(F\)|\(G\)|\(H\)|\(I\)|\(J\)', answer)
        if len(pred) == 1:
            mapping = {'(A)': 0, '(B)':1, '(C)': 2, '(D)': 3, '(E)': 4, '(F)': 5, '(G)': 6, '(H)': 7, '(I)': 8, '(J)': 9}
            return mapping[pred[0]]
        
        pred = re.search(r'A|B|C|D|E|F|G|H|I|J', answer)
        if len(pred) > 0:
            mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9}
            return mapping[pred[0]]
        return None


    def run(self, given_phrase, images):

        captions_list = [self.captioner.run(img) for img in images]
        llm_prompt = self.prompt(given_phrase=given_phrase, captions_list=captions_list, strategy=self.strategy)
        llm_answer = self.llm.completion(llm_prompt)
        if self.CoT_flag:
            llm_prompt = CoT_prompt(given_phrase=given_phrase, captions_list=captions_list, strategy=self.strategy, llm_response=llm_answer)
            llm_answer = self.llm.completion(llm_prompt)
        return self.parse_answer(llm_answer)



class FewShotQAModule:

    def __init__(self, llm, captioner, strategy, prompt_template, few_shot_selector):

        if llm not in AVAILABLE_LLMS:
            raise ValueError(f"Invalid LLM: {llm}. Should be one of {AVAILABLE_LLMS.keys()}")
        
        if captioner not in AVAILABLE_CAPTIONERS:
            raise ValueError(f"Invalid LLM: {captioner}. Should be one of {AVAILABLE_CAPTIONERS.keys()}")

        if strategy not in ['greedy', 'beam']:
            raise ValueError(f"Invalid captioner strategy: {strategy}. Should be one of ['greedy', 'beam']")
        
        if prompt_template not in AVAILABLE_FEW_SHOT_QA_PROMPT_TEMPLATES:
            raise ValueError(f"Invalid prompt template: {prompt_template}. Should be one of {AVAILABLE_FEW_SHOT_QA_PROMPT_TEMPLATES.keys()}")

        if not isinstance(few_shot_selector, FewShotSelectionModule):
            raise ValueError("Invalid Few Shot Selection Module")

        self.strategy = strategy
        self.llm = AVAILABLE_LLMS[llm]()
        self.captioner = AVAILABLE_CAPTIONERS[captioner](strategy)
        self.prompt = AVAILABLE_FEW_SHOT_QA_PROMPT_TEMPLATES[prompt_template]
        self.few_shot_selector = few_shot_selector

    def parse_answer(self, answer):
        pred = re.findall(r'\(A\)|\(B\)|\(C\)|\(D\)|\(E\)|\(F\)|\(G\)|\(H\)|\(I\)|\(J\)', answer)
        if len(pred) == 1:
            mapping = {'(A)': 0, '(B)':1, '(C)': 2, '(D)': 3, '(E)': 4, '(F)': 5, '(G)': 6, '(H)': 7, '(I)': 8, '(J)': 9}
            return mapping[pred[0]]
        
        pred = re.search(r'A|B|C|D|E|F|G|H|I|J', answer)
        if len(pred) > 0:
            mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9}
            return mapping[pred[0]]
        return None

    def run(self, given_phrase, images):

        few_shot_examles = self.few_shot_selector.select(test_given_phrase=given_phrase)

        prompt = ""
        # few shot examples 
        for example in few_shot_examles:
            captions_list = [self.captioner.run(img) for img in example['images']]
            prompt += self.prompt(given_phrase=example['given_phrase'], captions_list=captions_list, strategy=self.strategy, gold_image_index=example['gold_image_index'])
            prompt += "\n"
        # test example
        captions_list = [self.captioner.run(img) for img in images]
        prompt += self.prompt(given_phrase=given_phrase, captions_list=captions_list, strategy=self.strategy)
        llm_answer = self.llm.completion(prompt)
        return self.parse_answer(llm_answer)