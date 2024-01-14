
letters_list = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)', '(I)', '(J)']

def no_CoT_prompt(given_phrase, captions_list, strategy, gold_image_index=None):
    
    if gold_image_index is None:
        final_answer = ""
    else:
        # in few-shot setting
        if strategy == "greedy":
            final_answer = f"{letters_list[gold_image_index]} {captions_list[gold_image_index]}"
        elif strategy == "beam":
            final_answer = f"{letters_list[gold_image_index]} [{', '.join(captions_list[gold_image_index])}]"


    if strategy == "greedy":
        answer_choices = ', '.join([f"{letter} {caption}" for letter, caption in zip(letters_list, captions_list)])
        return f"What is the most appropriate caption for the {given_phrase}?\nAnswer Choices: {answer_choices}\nA:{final_answer}"
    elif strategy == "beam":
        answer_choices = ', '.join([f"{letter} [{', '.join(captions)}]" for letter, captions in zip(letters_list, captions_list)])
        return f"What is the most appropriate group of captions for the {given_phrase}?\nAnswer Choices: {answer_choices}\nA:{final_answer}"


def think_prompt(given_phrase, captions_list, strategy):
    return no_CoT_prompt(given_phrase, captions_list, strategy) + " Let\'s think step by step."

def CoT_prompt(given_phrase, captions_list, strategy, llm_response):
    return f"{think_prompt(given_phrase, captions_list, strategy)}\n{llm_response} Therefore, among A through J the answer is "


def choose_no_CoT_prompt(given_phrase, captions_list, strategy):
    if strategy == "greedy":
        answer_choices = '\n'.join([f"{letter} {caption}" for letter, caption in zip(letters_list, captions_list)])
        return (
            f"You have ten images, (A) to (J), which are given to you in the form of captions.\n{answer_choices}\n" + 
            f"You should choose the image, and therefore the caption that could better represent the {given_phrase}.\n" +
            "What image do you choose?"
        )
    elif strategy == "beam":
        answer_choices = ', '.join([f"{letter} [{', '.join(captions)}]" for letter, captions in zip(letters_list, captions_list)])
        return (
            f"You have ten images, (A) to (J), which are given to you in the form of captions.\n{answer_choices}\n" + 
            f"You should choose the image, and therefore the set of captions that could better represent the {given_phrase}.\n" +
            "What image do you choose?"
        )

def choose_CoT_prompt(given_phrase, captions_list, strategy):
    if strategy == "greedy":
        answer_choices = '\n'.join([f"{letter} {caption}" for letter, caption in zip(letters_list, captions_list)])
        return (
            f"You have ten images, (A) to (J), which are given to you in the form of captions.\n{answer_choices}\n" + 
            f"You should choose the image, and therefore the caption that could better represent the {given_phrase}.\n" +
            "Use the following format:\n" +
            "Question: What image do you choose?\n" +
            "Thought: you should always think about what you choose\n" +
            "Result: the result of your thought\n" +
            "Final Answer: the image you choose\n" +
            "Begin!\n" +
            "Question: What image do you choose?"
        )
    elif strategy == "beam":
        answer_choices = ', '.join([f"{letter} [{', '.join(captions)}]" for letter, captions in zip(letters_list, captions_list)])
        return (
            f"You have ten images, (A) to (J), which are given to you in the form of captions.\n{answer_choices}\n" + 
            f"You should choose the image, and therefore the set of captions that could better represent the {given_phrase}.\n" +
            "Use the following format:\n" +
            "Question: What image do you choose?\n" +
            "Thought: you should always think about what you choose\n" +
            "Result: the result of your thought\n" +
            "Final Answer: the image you choose\n" +
            "Begin!\n" +
            "Question: What image do you choose?"
        )
