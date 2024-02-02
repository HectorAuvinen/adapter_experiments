from datasets import load_dataset,DatasetDict,ClassLabel
import logging

GLUE_TASKS = ['ax', 'cola', 'mnli', 'mnli_matched', 'mnli_mismatched', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli']
SUPER_GLUE_TASKS = ['axb', 'axg', 'boolq', 'cb', 'copa', 'multirc', 'record', 'rte', 'wic', 'wsc', 'wsc.fixed']
class_label_bool = ClassLabel(num_classes=2,names=["False","True"])

logging.basicConfig(
    level=logging.INFO,  # Set the logging level as needed
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="my_log_file.log"  # Specify the log file name
)

def boolean_to_int_label(data):
    data["label"] = class_label_bool.str2int(str(data["label"]))
    return data

def load_hf_dataset(task_name:str,
                    debug:bool=False) -> DatasetDict:
    
    if task_name in GLUE_TASKS:
        dataset = load_dataset("glue",task_name)
    elif task_name in SUPER_GLUE_TASKS:
        dataset = load_dataset("super_glue",task_name)
    else:
        dataset = load_dataset(task_name)
        if task_name == "boolq":
            dataset = dataset.rename_column("answer","label")
            dataset = dataset.map(boolean_to_int_label)
            dataset = dataset.cast_column("label",class_label_bool)
            
    
    if debug:
        selected_datasets = {split: dataset[split].select(range(10)) for split in dataset.keys()}
        # Convert the dictionary back into a DatasetDict
        dataset = DatasetDict(selected_datasets)
    
    # Use the logger to replace print statements
    logging.info(f"Number of rows: {dataset.num_rows}")
    logging.info(f"Features for 'train' split: {dataset['train'].features}")
    logging.info(f"Sample from 'train' split: {dataset['train'][0]}")
    
    return dataset
    