from utils import process_labels

def encode_general_classification(data, tokenizer) -> dict:
    """Encode a batch of input data that is in the format text,label"""
    return tokenizer(data["text"],max_length=128,truncation=True,padding="max_length")

def encode_hellaswag(data,tokenizer) -> dict:
    pass

def encode_winogrande(data,tokenizer) -> dict:
    """Encodes a batch of input data using the model tokenizer."""
    all_encoded = {"input_ids": [], "attention_mask": [], "labels": []}

    processed_labels = process_labels(all_encoded["answer"])
    
    for sentence, option1,option2,label in zip(data["sentence"], data["option1"],data["option2"], processed_labels):
        sentences_a = [sentence for _ in range(2)]
        sentences_b = [option1,option2]
        encoded = tokenizer(
            sentences_a,
            sentences_b,
            max_length=128,  # or any max_length that suits your model/context
            truncation=True,
            padding="max_length",
        )
        all_encoded["input_ids"].append(encoded["input_ids"])
        all_encoded["attention_mask"].append(encoded["attention_mask"])
        all_encoded["labels"].append(label)
        # Add the labels. Convert label to int if it's not already (assuming label is the index of the correct ending)
        # labels currently 1,2 -> convert to 0,1
        #all_encoded["labels"].append(int(label)-1 if isinstance(label, str) and label.isdigit() else int(0)) # was else label

    return all_encoded

def encode_cosmosqa(data,tokenizer) -> dict:
    pass

def encode_socialiqa(data,tokenizer) -> dict:
    pass

def encode_imdb(data,tokenizer) -> dict:
    pass

def encode_sst2(data,tokenizer) -> dict:
    pass

def encode_mnli(data,tokenizer) -> dict:
    pass

def encode_sickr(data,tokenizer) -> dict:
    pass

def encode_rte(data,tokenizer) -> dict:
    pass

def encode_cb(data,tokenizer) -> dict:
    pass

def encode_mrpc(data,tokenizer) -> dict:
    pass

def encode_qqp(data,tokenizer) -> dict:
    pass

def encode_argumentmining(data,tokenizer) -> dict:
    pass

def encode_boolq(data,tokenizer) -> dict:
    pass
