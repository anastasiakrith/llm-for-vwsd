import numpy as np

from modules.vl_transformers import CLIP

AVAILABLE_PENALTY_MODELS = {
    'clip': lambda: CLIP(large=False)
}


class PenaltyModule:

    def __init__(self, dataset, penalty_model='clip'):
        
        if (penalty_model is not None) and (penalty_model not in AVAILABLE_PENALTY_MODELS):
            raise ValueError(f"Invalid Penalty model: {penalty}. Should be one of {AVAILABLE_PENALTY_MODELS.keys()}")

        self.penalty_model = AVAILABLE_PENALTY_MODELS[penalty_model]()
        self.dataset = dataset
    
    
    def penalty_factor(self, image, image_name):

        vl_response = self.penalty_model.run(self.dataset.given_phrases[:400], image)
        similarity_score = vl_response['similarity_score']
        mean_score = np.sum(similarity_score) / len(similarity_score)
        normalized_card = self.dataset.image_cardinality[image_name] / np.max(list(self.dataset.image_cardinality.values()))
        return mean_score * normalized_card