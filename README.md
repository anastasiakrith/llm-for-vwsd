# Language Models as Knowledge Bases for Visual Word Sense Disambiguation (VWSD)

## Install

```
git clone https://github.com/anastasiakrith/llm-for-vwsd.git
cd llm-for-vwsd
```

### Setting up (virtualenv)

On the project folder run the following commands:

1. ```$ virtualenv env```    to create a virtual environment
2. ```$ source venv/bin/activate``` to activate the environment
3. ```$ pip install -r requirements.txt``` to install packages
4. Create a ```.env``` file with the environmental variables. The project needs a ```OPENAI_API_KEY``` with the API key corresponding to your openai account, and optionally a ```DATASET_PATH``` corresponding to the absolute path of [VWSD dataset](https://raganato.github.io/vwsd/).


## Running the project

### VL Retrieval
```
python vl_retrieval_eval.py -llm "gpt-3" -vl "clip" -baseline -penalty 
```

### QA Retrieval
```
python qa_retrieval_eval.py -llm "gpt-3.5" -captioner "git" -strategy "greedy" -prompt "no_CoT" -zero_shot
```
