import torch
import open_clip

from transformers import CLIPProcessor, CLIPModel, AlignProcessor, AlignModel, BlipForImageTextRetrieval, AutoProcessor

class CLIP:
    """
    Wrapper for openai/clip-vit-{}-patch{} from hugging-face
    """
    def __init__(self, large=False):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.large = large
        if not self.large:
            self.load_from = "openai/clip-vit-base-patch32"
        else:
            self.load_from = "openai/clip-vit-large-patch14"
        self.model = CLIPModel.from_pretrained(self.load_from).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.load_from)
    
    def run(self, text, images):

        model_input = self.processor(text=text, images=images, return_tensors="pt", padding=True).to(self.device)
        self.model.eval()
        with torch.no_grad():
            model_output = self.model(**model_input)
        similarity_score = model_output.logits_per_image.flatten()
        probs = similarity_score.softmax(dim=0)
        
        return {'probs': probs.tolist(), 'similarity_score': similarity_score.tolist()}


class CLIP_LAION:
    """
    Wrapper for laion/CLIP-ViT-H-14-laion2B-s32B-b79K from hugging-face
    """
    def __init__(self):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_from = "hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        self.model, _, self.processor = open_clip.create_model_and_transforms(self.load_from)
        self.model = self.model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer(self.load_from)

    def run(self, text, images):

        images_features = [self.processor(img).unsqueeze(0) for img in images]
        images_features = torch.cat(images_features).to(self.device)
        text_features = self.tokenizer(text).to(self.device)

        self.model.eval()
        with torch.no_grad():
            image_features = self.model.encode_image(images_features)
            text_features = self.model.encode_text(text_features)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            logits_per_image = 100.0 * text_features @ image_features.T
            probs = (100.0 * text_features @ image_features.T).softmax(dim=-1).flatten()

            return {'probs': probs.tolist(), 'similarity_score': logits_per_image.flatten().tolist()}
        

class ALIGN:
    """
    Wrapper for kakaobrain/align-base from hugging-face
    """

    def __init__(self):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_from = "kakaobrain/align-base"
        self.model = AlignModel.from_pretrained(self.load_from).to(self.device)
        self.processor = AlignProcessor.from_pretrained(self.load_from)

    def run(self, text, images):

        model_input = self.processor(text=text, images=images, return_tensors="pt").to(self.device)
        self.model.eval()
        with torch.no_grad():
            model_output = self.model(**model_input)
        similarity_score = model_output.logits_per_image.flatten()
        probs = similarity_score.softmax(dim=0)
        
        return {'probs': probs.tolist(), 'similarity_score': similarity_score.tolist()}


class BLIP_COCO:
    """
    Wrapper for Salesforce/blip-itm-{}-coco from hugging-face
    """
    def __init__(self, large=False):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.large = large
        if not self.large:
            self.load_from = "Salesforce/blip-itm-base-coco"
        else:
            self.load_from = "Salesforce/blip-itm-large-coco"
        self.model = BlipForImageTextRetrieval.from_pretrained(self.load_from).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.load_from)
    
    def run(self, text, images):

        model_input = self.processor(text=text, images=images, return_tensors="pt").to(self.device)
        self.model.eval()
        with torch.no_grad():
            model_output = self.model(**model_input)
        similarity_score = model_output.itm_score[:, 1].flatten()
        probs = similarity_score.softmax(dim=0).flatten()
        
        return  {'probs': probs.tolist(), 'similarity_score': similarity_score.tolist()}


class BLIP_FLICKR:
    """
    Wrapper for Salesforce/blip-itm-{}}-flickr from hugging-face
    """
    def __init__(self, large=False):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.large = large
        if not self.large:
            self.load_from = "Salesforce/blip-itm-base-flickr"
        else:
            self.load_from = "Salesforce/blip-itm-large-flickr"
        self.model = BlipForImageTextRetrieval.from_pretrained(self.load_from).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.load_from)
    
    def run(self, text, images):

        model_input = self.processor(text=text, images=images, return_tensors="pt").to(self.device)
        self.model.eval()
        with torch.no_grad():
            model_output = self.model(**model_input)
        similarity_score = model_output.itm_score[:, 1].flatten()
        probs = similarity_score.softmax(dim=0).flatten()
        
        return  {'probs': probs.tolist(), 'similarity_score': similarity_score.tolist()}
