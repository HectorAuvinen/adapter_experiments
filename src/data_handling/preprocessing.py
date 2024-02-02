from .utils import process_labels
from transformers import BertTokenizer
    


def get_tokenizer(model_name):
    if "bert" in model_name:
        print("using bert tokenizer")
        return BertTokenizer.from_pretrained(model_name)
    print("not using bert tokenizer")
    
def map_clf_dataset(dataset, encode:callable):
    dataset = dataset.map(encode,batched=True)
    dataset = dataset.rename_column(original_column_name="label",new_column_name="labels")
    dataset.set_format(type="torch",columns=["input_ids","attention_mask","labels"])
    return dataset

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
    """Encodes a batch of input data using the model tokenizer."""
    all_encoded = {"input_ids": [], "attention_mask": [], "labels": []}

    for sentence, answer0,answer1,answer2,answer3,label in zip(data["context"], data["answer0"],data["answer1"],data["answer2"],data["answer3"], data["label"]):
        sentences_a = [sentence for _ in range(4)]
        sentences_b = [answer0, answer1, answer2, answer3]
        encoded = tokenizer(
            sentences_a,
            sentences_b,
            max_length=256,  # or any max_length that suits your model/context
            truncation=True,
            padding="max_length",
        )
        all_encoded["input_ids"].append(encoded["input_ids"])
        all_encoded["attention_mask"].append(encoded["attention_mask"])
        all_encoded["labels"].append(label)

    return all_encoded

def encode_socialiqa(data,tokenizer) -> dict:
    all_encoded = {"input_ids":[],"attention_mask":[]}
    for context,question,answera,answerb,answerc in zip(data["context"],data["question"],data["answerA"],data["answerB"],data["answerC"]):
        sentences_a = [context + " " + question for _ in range(3)]
        sentences_b = [answera,answerb,answerc]
        encoded = tokenizer(
            sentences_a,
            sentences_b,
            max_length=128,
            truncation=True,
            padding="max_length"
        )
        all_encoded["input_ids"].append(encoded["input_ids"])
        all_encoded["attention_mask"].append(encoded["attention_mask"])
    
    return all_encoded

def encode_imdb(data,tokenizer,config) -> dict:
    pass

def encode_sst2(data,tokenizer) -> dict:
    return tokenizer(data["sentence"],max_length=128,truncation=True,padding="max_length")

def encode_mnli(data,tokenizer) -> dict:
    pass

def encode_sick(data,tokenizer) -> dict:
    return tokenizer(data["sentence_A"],data["sentence_B"],max_length=180,truncation=True,padding="max_length")

def encode_rte(data,tokenizer) -> dict:
    return tokenizer(data["sentence1"],data["sentence2"],max_length=180,truncation=True,padding="max_length")

def encode_cb(data,tokenizer) -> dict:
    return(tokenizer(data["premise"],data["hypothesis"],max_length=180,truncation=True,padding="max_length"))

def encode_mrpc(data,tokenizer) -> dict:
    return tokenizer(data["sentence1"],data["sentence2"],max_length=180,truncation=True,padding="max_length")

def encode_qqp(data,tokenizer) -> dict:
    pass

def encode_argumentmining(data,tokenizer) -> dict:
    pass

def encode_boolq(data,tokenizer) -> dict:
    return tokenizer(data["question"],data["passage"],max_length=128,truncation=True,padding="max_length",return_overflowing_tokens=True)

def encode_wrapper(data, tokenizer, encoding_func, **kwargs):
    def wrapper(batch):
        return encoding_func(batch, tokenizer, **kwargs)
    return wrapper

def preprocess_dataset(dataset,encoding_func,tokenizer):
    # Encode the input data
    dataset = dataset.map(encode_wrapper(dataset,tokenizer,encoding_func), batched=True)
    print("mapped")
    # The transformers model expects the target class column to be named "labels"
    # Check if renaming is necessary based on your dataset structure
    print(dataset.column_names)
    dataset = dataset.rename_column("label", "labels")
    # Transform to pytorch tensors and only output the required columns
    dataset.set_format(columns=["input_ids", "attention_mask", "labels"])
    return dataset   


encode_map = {
    "sst2":encode_sst2,
    "winogrande":encode_winogrande,
    "cb":encode_cb,
    "rte":encode_rte,
    "sick":encode_sick,
    "mrpc":encode_mrpc,
    "boolq":encode_boolq,
    
}

def get_encoding(task_name):
    print("getting encoding:")
    print(encode_map[task_name])
    return encode_map[task_name]

def get_label_count(dataset):
    id2label = {id: label for (id,label) in enumerate(dataset["train"].features["labels"].names)}
    num_labels = len(id2label)
    return num_labels