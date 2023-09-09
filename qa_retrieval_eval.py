import argparse
from tqdm import tqdm

from modules.qa import ZeroShotQAModule
from modules.qa import FewShotQAModule
from modules.qa import AVAILABLE_LLMS, AVAILABLE_CAPTIONERS, AVAILABLE_ZERO_SHOT_QA_PROMPT_TEMPLATES, AVAILABLE_FEW_SHOT_QA_PROMPT_TEMPLATES
from modules.few_shot_selection import FewShotSelectionModule
from modules.dataset import Dataset
from modules.metrics import ScoreModule

def get_k_default_value(captioner, captioner_strategy):
    if captioner_strategy == 'greedy': # all greedy
        return 5
    if captioner == 'blip-large': # blip-large beam
        return 1
    return 2 # git-large / vit-gpt2 beam


def run_qa_retrieval(llm, captioner, captioner_strategy, prompt, few_shot_strategy, zero_shot, dataset_path):

    dataset = Dataset(base_dir=dataset_path)

    test_dataset = dataset.test_dataloader()

    if zero_shot:
        qa = ZeroShotQAModule(llm=llm, captioner=captioner, strategy=captioner_strategy, prompt_template=prompt)
    else:
        train_dataset = dataset.train_dataloader()
        few_shot_selector = FewShotSelectionModule(strategy=few_shot_strategy, train_dataset=train_dataset, k=get_k_default_value(captioner=captioner, captioner_strategy=captioner_strategy))
        qa = FewShotQAModule(llm=llm, captioner=captioner, strategy=captioner_strategy, prompt_template=prompt, few_shot_selector=few_shot_selector)

    score = ScoreModule(approach='qa_retrieval')        
    for i in tqdm(range(len(test_dataset))):
        prediction = qa.run(given_phrase=test_dataset[i]['given_phrase'], images=test_dataset[i]['images'])
        score.add(golden_image_index=test_dataset[i]['gold_image_index'], predictions=prediction)

    print(f'Accuracy Score: {score.accuracy_score()}')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-llm", "--llm_model", default=None, help=f"Choose LLM model. Options: {AVAILABLE_LLMS.keys()}")
    parser.add_argument("-captioner", "--captioner", default="git", help=f"Choose Captioner model. Options: {AVAILABLE_CAPTIONERS.keys()}")
    parser.add_argument("-strategy", "--captioner_strategy", default="greedy", help=f"Choose Captioner strategy. Options: 'greedy', 'beam'")
    parser.add_argument("-prompt", "--prompt", default='no_CoT', help=f"Choose prompt. Options: {AVAILABLE_ZERO_SHOT_QA_PROMPT_TEMPLATES.keys()}")
    parser.add_argument("-few_shot", "--few_shot_strategy", default=None, help=f"Use few shot setting and choose strategy. Options: 'random', 'top', 'inverse_top'")
    parser.add_argument("-zero_shot", "--zero_shot", action="store_true", help="Use zero shot setting.")
    parser.add_argument("-dataset_path", "--dataset_path", default=None, help=f"Set dataset path.")

    args = parser.parse_args()

    run_qa_retrieval(
        llm=args.llm_model,
        captioner=args.captioner,
        captioner_strategy=args.captioner_strategy,
        prompt=args.prompt,
        few_shot_strategy=args.few_shot_strategy,
        zero_shot=args.zero_shot,
        dataset_path=args.dataset_path
    )
