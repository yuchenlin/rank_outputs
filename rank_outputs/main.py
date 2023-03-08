from transformers import AutoTokenizer, AutoConfig
from utils import ModelBase, DataCollatorForMultipleChoice, convert_features
from torch.utils.data import DataLoader
import torch


def init_model(model_name_or_path="google/flan-t5-small"):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    config = AutoConfig.from_pretrained(model_name_or_path)
    model = ModelBase.from_config(
            config=config,
            model_name_or_path=model_name_or_path,
            parallelize=False
        )
    return model, tokenizer

def get_example_data():
    input_texts = [
        "Answer this question: I am black when you buy me, red when you use me. When I turn white, you know it's time to trow me away. What am I? ",
        "Answer this question: I have a long tail that I let fly. Every time I go through a gap, I leave a bit of my tail in the trap. What am I?", 
        "Compose a sentence that contains all given concepts: book read boy chair sit", 
    ]    
    answer_choices_texts = [
        ["rose flower", "ink", "charcoal", "fruit", "shoe"],
        ["monkey", "basketball", "fishing pole", "comet asdf", "needle"],
        ["A boy read chair that sits on a book.", "A boy sits on a chair and read a book.", "A book reads a boy that sits on a chair."], # "placeholder", "placeholder 1"] 
                        # if the number of choices is different, please either use bsz=1 or add some placeholders to make them the same 
    ]
    target_texts = ["charcoal", "N/A", "A boy sits on a chair and read a book."] # optional, will not be used for ranking but maybe useful for evaluation
    
    data = []
    for i in range(len(input_texts)):
        item = {"input_texts": [input_texts[i]], "answer_choices_texts": [answer_choices_texts[i]], "target_texts": [target_texts[i]]}
        data.append(item)
    return data

def main():
    model, tokenizer = init_model("google/flan-t5-small")
    # model, tokenizer = init_model("google/flan-t5-base")
    # model, tokenizer = init_model("facebook/bart-base") # TODO(yuchenl): not work yet  
    # model, tokenizer = init_model("bigscience/bloom-560m") # TODO(yuchenl): not work yet  
    
    data = get_example_data()

    
    features = [convert_features(tokenizer, item) for item in data]

    print(features[0]["labels_attention_mask"])
 
    
    data_collator = DataCollatorForMultipleChoice(
                tokenizer, pad_to_multiple_of=None, padding=True, max_length=64
                )
    eval_dataloader = DataLoader(features, collate_fn=data_collator, batch_size=1)
    model.eval()
 

    for batch in eval_dataloader: 
        with torch.no_grad():
            predictions, seq_log_prob = model(batch)
            print(predictions) 
            print(seq_log_prob) 

main()

""""
Example output (for t5-small): 

tensor([1])
tensor([[-30.0734, -12.5917, -18.3455, -19.3478, -13.3692]])
tensor([4])
tensor([[-16.2935, -19.1336, -15.2122, -42.6995, -12.8641]])
tensor([1])
tensor([[-36.2724, -24.8918, -30.3378]])

"""