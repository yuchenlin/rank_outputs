from transformers import AutoTokenizer, AutoConfig
from utils import ModelBase, DataCollatorForMultipleChoice
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


def convert_features(tokenizer, data): 
    input_texts = data["input_texts"]
    answer_choices_texts = data["answer_choices_texts"]
    target_texts = data["target_texts"]
    bs = len(input_texts)
    padding = "max_length" 
    max_length = 64
    
    tokenized_inputs = tokenizer(
                input_texts,
                padding=padding,
                max_length=max_length,
                truncation=True,
                add_special_tokens=False,
            )

    tokenized_targets = [
        tokenizer(
            options,
            # padding is on the right here.
            padding=padding,
            max_length=max_length,
            truncation=True,
        )
        for options in answer_choices_texts
    ]
    

    features = {
                k: [
                    [elem for _ in range(len(tokenized_targets[idx]["input_ids"]))]
                    for idx, elem in enumerate(v)
                ]
                for k, v in tokenized_inputs.items()
            }

    features["labels"] = [
        tokenized_targets[idx]["input_ids"]
        for idx in range(bs)
    ]
    features["labels_attention_mask"] = [
        tokenized_targets[idx]["attention_mask"]
        for idx in range(bs)
    ]
    features["targets"] = [
        answer_choices_texts[idx].index(t) if t else -1
        for idx, t in enumerate(target_texts)
    ]
    return features

def get_example_data():
    input_texts = [
        "Answer this question: I am black when you buy me, red when you use me. When I turn white, you know it's time to trow me away. What am I? ",
        "Answer this question: I have a long tail that I let fly. Every time I go through a gap, I leave a bit of my tail in the trap. What am I?", 
        "Compose a sentence that contains all given concepts: book read boy chair sit", 
    ]    
    answer_choices_texts = [
        ["rose flower", "ink", "charcoal", "fruit", "shoe"],
        ["monkey", "basketball", "fishing pole", "comet asdf", "needle"],
        ["A boy read chair that sits on a book.", "A boy sits on a chair and read a book.", "A book reads a boy that sits on a chair.", "placeholder", "placeholder 2"],
    ]
    target_texts = ["charcoal", "needle", "A boy sits on a chair and read a book."] # optional 
    
    return {"input_texts": input_texts, "answer_choices_texts": answer_choices_texts, "target_texts": target_texts}

def main():
    model, tokenizer = init_model("google/flan-t5-small")
    data = get_example_data()

    
    features = convert_features(tokenizer, data)
    # print(len(features["input_ids"]), len(features["labels"]))
    
    data_collator = DataCollatorForMultipleChoice(
                tokenizer, pad_to_multiple_of=None, padding=True, max_length=64
                )
    eval_dataloader = DataLoader([features], collate_fn=data_collator, batch_size=1)
    model.eval()

    # features = [features]
    # batch = data_fomratter(tokenizer, features) 

    for batch in eval_dataloader:
        # for k, v in batch.items():
        #     print(k, v.shape)
        batch = {
            k: v.view(v.shape[0]*v.shape[1], v.shape[2]) if k!="targets" else v.view(v.shape[1])
            for k, v in batch.items()
        }
        with torch.no_grad():
            predictions, seq_log_prob = model(batch)
            print(predictions) 

main()