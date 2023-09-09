import faiss
import torch 
import numpy as np 
from transformers import AlignProcessor, AlignModel


class AlignTextEmbeddingModel:

    def __init__(self):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_from = "kakaobrain/align-base"
        self.model = AlignModel.from_pretrained(self.load_from).to(self.device)
        self.processor = AlignProcessor.from_pretrained(self.load_from)

    def get_embedding(self, text):

        model_input = self.processor(text=text, return_tensors="pt").to(self.device)
        self.model.eval()
        with torch.no_grad():
            model_output = self.model.get_text_features(**model_input)
        
        return model_output.cpu().flatten().numpy() 



AVAILABLE_EMBEDDING_MODELS = {
    'align': lambda: AlignTextEmbeddingModel(),
}


class FewShotSelectionModule:

    def __init__(self, train_dataset, strategy, k=None, embedding_model='align'):
        
        if strategy not in ['random', 'top', 'inverse_top']:
            raise ValueError(f"Invalid few shot strategy: {strategy}. Should be one of ['random', 'top', 'inverse_top']")
        
        self.strategy = strategy
        self.k = k
        self.train_dataset = train_dataset
        if self.strategy != 'random' and embedding_model not in AVAILABLE_EMBEDDING_MODELS:
            raise ValueError(f"Invalid VL transformer: {embedding_model}. Should be one of {AVAILABLE_EMBEDDING_MODELS.keys()}")

        if self.strategy != 'random':
            self.embedding_model = AVAILABLE_EMBEDDING_MODELS[embedding_model]()
            self.create_faiss_vectorstore()

    
    def create_faiss_vectorstore(self):

        embeddings = np.array([self.embedding_model.get_embedding(given_phrase) for given_phrase in self.train_dataset.given_phrases])

        self.vectorstore = faiss.IndexFlatL2(embeddings[0].shape[0])
        faiss.normalize_L2(embeddings)
        self.vectorstore.add(embeddings)


    def random_selection(self):
        few_shot_examles = []
        for idx in np.random.randint(0, high=len(self.train_dataset), size=int(self.k)):
            train_sample = self.train_dataset[idx]
            few_shot_examles.append({
                'given_phrase': train_sample['given_phrase'],
                'images': train_sample['images'],
                'gold_image_index': train_sample['gold_image_index']
            })
        return few_shot_examles


    def top_selection(self, test_given_phrase, inverse=False):

        test_embedding = self.embedding_model.get_embedding(test_given_phrase)
        test_vector = np.expand_dims(test_embedding, axis=0)
        faiss.normalize_L2(test_vector)
        distances, ann = self.vectorstore.search(test_vector, k=self.k)

        few_shot_examles = []
        for idx in ann.flatten() if not inverse else ann.flatten()[::-1]:
            train_sample = self.train_dataset[idx]
            few_shot_examles.append({
                'given_phrase': train_sample['given_phrase'],
                'images': train_sample['images'],
                'gold_image_index': train_sample['gold_image_index']
            })
        return few_shot_examles


    def select(self, test_given_phrase=None):
        if self.strategy == 'random':
            return self.random_selection()
        if self.strategy == 'top':
            return self.top_selection(test_given_phrase, inverse=False)
        if self.strategy == 'inverse_top':
            return self.top_selection(test_given_phrase, inverse=True)