import argparse
from tqdm import tqdm

from modules.vl_retrieval import VLRetrievalModule
from modules.penalty import PenaltyModule
from modules.vl_retrieval import AVAILABLE_LLMS, AVAILABLE_VL_TRANSFORMERS, AVAILABLE_VL_PROMPT_TEMPLATES
from modules.dataset import Dataset
from modules.metrics import ScoreModule

def run_vl_retrieval(llm, vl_transformer, prompt, baseline, penalty, dataset_path=None):

    dataset = Dataset(base_dir=dataset_path)
    test_dataset = dataset.test_dataloader()

    if penalty:
        penalty_module = PenaltyModule(dataset=test_dataset)        
    else:
        penalty_module = None
    
    vl_retrieval = VLRetrievalModule(vl_transformer=vl_transformer, llm=llm, prompt_template=prompt, baseline=baseline, penalty=penalty_module)

    score = ScoreModule(approach='vl_retrieval')           
    for i in tqdm(range(len(test_dataset))):
        retrieval = vl_retrieval.run(given_phrase=test_dataset[i]['given_phrase'], images=test_dataset[i]['images'], images_names=test_dataset[i]['images'])
        score.add(golden_image_index=test_dataset[i]['gold_image_index'], predictions=retrieval['ordered_pred_images'])

    print(f'Accuracy Score: {score.accuracy_score()}')
    print(f'MRR Score: {score.mrr_score()}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-llm", "--llm_model", default=None, help=f"Choose LLM model. Options: {AVAILABLE_LLMS.keys()} or None")
    parser.add_argument("-vl", "--vl_transformer", default="clip", help=f"Choose VL transformer model. Options: {AVAILABLE_VL_TRANSFORMERS.keys()}")
    parser.add_argument("-prompt", "--prompt", default=None, help=f"Choose prompt. Options: {AVAILABLE_VL_PROMPT_TEMPLATES.keys()} or None")
    parser.add_argument("-baseline", "--baseline", action="store_true", help="Baseline")
    parser.add_argument("-penalty", "--penalty", action="store_true", help="Add penalty factor")
    parser.add_argument("-dataset_path", "--dataset_path", default=None, help=f"Set dataset path.")

    args = parser.parse_args()

    run_vl_retrieval(
        llm=args.llm_model,
        vl_transformer=args.vl_transformer,
        prompt=args.prompt,
        baseline=args.baseline,
        penalty=args.penalty,
        dataset_path=args.dataset_path
    )
