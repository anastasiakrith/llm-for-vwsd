import numpy as np

class ScoreModule:
    def __init__(self, approach):
        self.approach = approach
        if self.approach == 'vl_retrieval':
            self.prediction_list = {i: 0 for i in range(10)}
        elif self.approach == 'qa_retrieval':
            self.correct_cnt = 0
        self.total_cnt = 0
    
    def add(self, golden_image_index, predictions):
        if self.approach == 'vl_retrieval':
            pred = np.where(np.array(predictions)==golden_image_index)[0][0]
            self.prediction_list[pred] += 1
        elif self.approach == 'qa_retrieval':
            if (predictions is not None) and gold_image_index == int(predictions):
                self.correct_cnt += 1
        
        self.total_cnt += 1
    
    def accuracy_score(self):
        if self.approach == 'vl_retrieval':
            return self.prediction_list[0] / self.total_cnt
        elif self.approach == 'qa_retrieval':
            return self.correct_cnt / self.total_cnt

    def mrr_score(self):
        if self.approach == 'vl_retrieval':
            mrr = np.sum([self.prediction_list[i] * (1/(i+1)) for i in range(10)])
            return mrr / self.total_cnt
        elif self.approach == 'qa_retrieval':
            raise ValueError("MRR Score is not supported for QA Retrieval")