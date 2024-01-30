from utils import process_labels


def encode_winogrande(data,tokenizer) -> dict:
    """Encodes a batch of input data using the model tokenizer."""
    all_encoded = {"input_ids": [], "attention_mask": [], "labels": []}
    #
    processed_labels = process_labels(all_encoded["answer"])
    
    #
    # Iterate through all examples in this batch
    #
    for sentence, option1,option2,label in zip(data["sentence"], data["option1"],data["option2"], data["answer"]):
        # 
        sentences_a = [sentence for _ in range(2)]
        sentences_b = [option1,option2]
        #print(sentences_b)
        encoded = tokenizer(
            sentences_a,
            sentences_b,
            max_length=128,  # or any max_length that suits your model/context
            truncation=True,
            padding="max_length",
        )
        all_encoded["input_ids"].append(encoded["input_ids"])
        all_encoded["attention_mask"].append(encoded["attention_mask"])
        # Add the labels. Convert label to int if it's not already (assuming label is the index of the correct ending)
        # labels currently 1,2 -> convert to 0,1
        #all_encoded["labels"].append(int(label)-1 if isinstance(label, str) and label.isdigit() else int(0)) # was else label

    return all_encoded