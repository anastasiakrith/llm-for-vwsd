import torch 
from transformers import AutoProcessor, BlipForConditionalGeneration, AutoModelForCausalLM, VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer


def clean_captions(captions):

    captions_set = set()
    captions = np.array(captions)

    len_captions = [len(x) for x in captions]
    indices = np.argsort(len_captions)[::-1]
    captions = captions[indices]
    
    for cap in captions:
        if not (cap in captions_set):
            flag = False
            for past_cap in captions_set:
                if cap in past_cap: # if current caption is substring of another one already in captions_set continue
                    flag = True
            if not flag:
                captions_set.add(cap)
    return list(captions_set)


class BLIP:
    """
    Wrapper for Salesforce/blip-image-captioning-{} from hugging-face
    """

    def __init__(self, strategy, large=False, max_tokens=70):

        self.strategy = strategy
        if self.strategy not in ["greedy", "beam"]:
            raise ValueError("Invalid strategy %s . Should be one of ['greedy', 'beam']", self.strategy)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.large = large
        self.max_tokens = max_tokens
        if not self.large:
            self.load_from = "Salesforce/blip-image-captioning-base"
        else:
            self.load_from = "Salesforce/blip-image-captioning-large"
        self.model = BlipForConditionalGeneration.from_pretrained(self.load_from).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.load_from)
    
    def run(self, image):
        model_input = self.processor(images=image, return_tensors="pt").to(self.device)
        self.model.eval()
        with torch.no_grad():
            if self.strategy == "greedy":
                model_output = self.model.generate(**model_input, max_new_tokens=self.max_tokens)
            elif self.strategy == "beam":
                model_output = self.model.generate(**model_input, max_new_tokens=self.max_tokens, num_return_sequences=10, num_beams=10, do_sample=True)
        
        if self.strategy == "greedy":
            return self.processor.decode(model_output[0], skip_special_tokens=True).strip()
        elif self.strategy == "beam":
            return clean_captions([self.processor.decode(output, skip_special_tokens=True).strip() for output in model_output])

        
        
class GIT:
    """
    Wrapper for microsoft/git-{}-coco from hugging-face
    """

    def __init__(self, strategy, large=False, max_tokens=70):

        self.strategy = strategy
        if self.strategy not in ["greedy", "beam"]:
            raise ValueError("Invalid strategy %s . Should be one of ['greedy', 'beam']", self.strategy)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.large = large
        self.max_tokens = max_tokens
        if not self.large:
            self.load_from = "microsoft/git-base-coco"
        else:
            self.load_from = "microsoft/git-large-coco"
        self.model = AutoModelForCausalLM.from_pretrained(self.load_from).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.load_from)
    

    def run(self, image):
        model_input = self.processor(images=image, return_tensors="pt").to(self.device)
        self.model.eval()
        with torch.no_grad():
            if self.strategy == "greedy":
                model_output = self.model.generate(pixel_values=model_input.pixel_values, max_new_tokens=self.max_tokens)
            elif self.strategy == "beam":
                model_output = self.model.generate(pixel_values=model_input.pixel_values, max_new_tokens=self.max_tokens, num_return_sequences=10, num_beams=10, do_sample=True)
        
        if self.strategy == "greedy":
            return self.processor.batch_decode(model_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].strip()
        elif self.strategy == "beam":
            return clean_captions([caption.strip() for caption in self.processor.batch_decode(model_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)])
        
    
class VIT_GPT2:
    """
    Wrapper for nlpconnect/vit-gpt2-image-captioning from hugging-face
    """

    def __init__(self, strategy, max_tokens=70):

        self.strategy = strategy
        if self.strategy not in ["greedy", "beam"]:
            raise ValueError("Invalid strategy %s . Should be one of ['greedy', 'beam']", self.strategy)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_tokens = max_tokens
        self.load_from = "nlpconnect/vit-gpt2-image-captioning"

        self.model = VisionEncoderDecoderModel.from_pretrained(self.load_from).to(self.device)
        self.processor = ViTImageProcessor.from_pretrained(self.load_from)
        self.tokenizer = AutoTokenizer.from_pretrained(self.load_from)
    

    def run(self, image):
        model_input = self.processor(images=image, return_tensors="pt").to(self.device)
        self.model.eval()
        with torch.no_grad():
            if self.strategy == "greedy":
                model_output = self.model.generate(pixel_values=model_input.pixel_values, max_new_tokens=self.max_tokens)
            elif self.strategy == "beam":
                model_output = self.model.generate(pixel_values=model_input.pixel_values, max_new_tokens=self.max_tokens, num_return_sequences=10, num_beams=10, do_sample=True)
        
        if self.strategy == "greedy":
            return self.tokenizer.batch_decode(model_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].strip()
        elif self.strategy == "beam":
            return clean_captions([caption.strip() for caption in self.tokenizer.batch_decode(model_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)])
