import numpy as np

from modules.vl_transformers import CLIP, CLIP_LAION, ALIGN, BLIP_COCO, BLIP_FLICKR
from modules.llms import GPT_3, GPT_3_5, BLOOMZ, OPT, VICUNA, LLAMA
from modules.penalty import PenaltyModule

AVAILABLE_VL_TRANSFORMERS = {
    'clip': lambda: CLIP(large=False),
    'clip_large': lambda: CLIP(large=True),
    'clip_laion': lambda: CLIP_LAION(),
    'align': lambda: ALIGN(),
    'blip_coco': lambda: BLIP_COCO(large=False),
    'blip_coco_large': lambda: BLIP_COCO(large=True),
    'blip_flickr': lambda: BLIP_FLICKR(large=False),
    'blip_flickr_large': lambda: BLIP_FLICKR(large=True)
}

AVAILABLE_LLMS = {
    'gpt-3': lambda: GPT_3(),
    'gpt-3.5': lambda: GPT_3_5(),
    'bloomz-1b7': lambda: BLOOMZ(nparameters='1b7'),
    'bloomz-3b': lambda: BLOOMZ(nparameters='3b'),
    'opt-2.7b': lambda: OPT(nparameters='2.7b'),
    'opt-6.7b': lambda: OPT(nparameters='6.7b'),
    'vicuna-7b': lambda: VICUNA(nparameters='7B'),
    'vicuna-13b': lambda: VICUNA(nparameters='13B'),
    'llama': lambda: LLAMA()
}


AVAILABLE_VL_PROMPT_TEMPLATES = {
    'exact': lambda x: f"{x} ",
    'what_is': lambda x: f"What is {x}?",
    'meaning_of': lambda x: f"What is the meaning of {x}?",
    'describe': lambda x: f"Describe {x}."
}

    
class VLRetrievalModule:

    def __init__(self, vl_transformer, llm, prompt_template=None, baseline=True, penalty=None):

        if vl_transformer not in AVAILABLE_VL_TRANSFORMERS:
            raise ValueError(f"Invalid VL transformer: {vl_transformer}. Should be one of {AVAILABLE_VL_TRANSFORMERS.keys()}")
        
        if llm not in AVAILABLE_LLMS:
            raise ValueError(f"Invalid LLM: {llm}. Should be one of {AVAILABLE_LLMS.keys()}")
        
        self.baseline = baseline
        if (not self.baseline) and (prompt_template not in AVAILABLE_VL_PROMPT_TEMPLATES):
            raise ValueError(f"Invalid Prompt Template: {prompt_template}. Should be one of {AVAILABLE_VL_PROMPT_TEMPLATES.keys()}")

        if (penalty is not None) and (not isinstance(penalty, PenaltyModule)):
            raise ValueError(f"Invalid Penalty model.")


        self.vl_transformer = AVAILABLE_VL_TRANSFORMERS[vl_transformer]()
        self.llm = AVAILABLE_LLMS[llm]()
        if not self.baseline:
            self.prompt = AVAILABLE_VL_PROMPT_TEMPLATES[prompt_template] 
       
        self.add_penalty = penalty is not None
        self.penalty_module = penalty


    def run(self, given_phrase, images, images_names):

        if self.baseline:
            text = given_phrase
        else:
            text = self.llm.completion(self.prompt(given_phrase))

        try:
            vl_response = self.vl_transformer.run(text, images)
            if not self.add_penalty:
                return {'ordered_pred_images': list(np.argsort(vl_response['probs'])[::-1])}    
            
            # Add penalty factor
            penalty_factors = [self.penalty_module.penalty_factor(img, img_name) for img, img_name in zip(images, images_names)]
            final_scores = np.array([sim - penalty for sim, penalty in zip(vl_response['similarity_score'], penalty_factors)])
            probalbilities = np.exp(final_scores) / sum(np.exp(final_scores)) # softmax
            return {'ordered_pred_images': list(np.argsort(probalbilities)[::-1])}  
        except: 
            return None