from datasets import load_dataset,DatasetDict,ClassLabel,load_from_disk
import os
import sys
import logging


script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the absolute path to the root of the project
project_root = os.path.dirname(script_dir)

# Add the project root to the sys.path
sys.path.insert(0, project_root)


from src.constants.constants import *


GLUE_TASKS = ['ax', 'cola', 'mnli', 'mnli_matched', 'mnli_mismatched', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli']
SUPER_GLUE_TASKS = ['axb', 'axg', 'boolq', 'cb', 'copa', 'multirc', 'record', 'rte', 'wic', 'wsc', 'wsc.fixed']
DISK_TASKS = {"argument":"C:/Users/Hector Auvinen/Desktop/UKP_sentential_argument_mining/hf_data/argument_mining"}

# class_label_bool = ClassLabel(num_classes=2,names=["False","True"])

logging.basicConfig(
    level=logging.INFO,  # Set the logging level as needed
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="my_log_file.log"  # Specify the log file name
)

#def boolean_to_int_label(data):
#    data["label"] = class_label_bool.str2int(str(data["label"]))
#    return data


def handle_disk_task(task_name):
    """
    Load and preprocess dataset from disk

    Args:
        task_name (str): task name to be used for loading the dataset
    Returns:
        dataset (DatasetDict): processed dataset
    """
    dataset = load_from_disk(DISK_TASKS[task_name])
    if task_name == "argument":
        dataset = dataset.rename_column(original_column_name="annotation",new_column_name="label")
        dataset = dataset.class_encode_column("label")
        num_labels = 3  
        class_label_feature = ClassLabel(num_classes=num_labels, names=['Argument_against', 'Argument_for', 'NoArgument'])
        dataset = dataset.cast_column('label', class_label_feature)
    return dataset

def load_scitail(task_name,format="tsv_format"):
    dataset = load_dataset(task_name,format)
    dataset = dataset.class_encode_column("label")
    class_label_feature = ClassLabel(num_classes=2, names=['entails', 'neutral'])
    dataset = dataset.cast_column('label', class_label_feature)
    return dataset

def load_imdb(task_name):
    dataset = load_dataset(task_name)
    dataset["validation"] = dataset.pop("test")
    return dataset

def load_winogrande(task_name,size="winograndle_xl"):
    dataset = load_dataset(task_name,size)
    dataset = dataset.rename_column("answer","label")
    dataset = dataset.class_encode_column("label")
    class_label_feature = ClassLabel(num_classes=2, names=['option1','option2'])
    dataset = dataset.cast_column('label', class_label_feature)
    return dataset

def process_boolq(dataset):
    # dataset = load_dataset(task_name)
    dataset = dataset.rename_column("answer","label")
    dataset = dataset.class_encode_column("label")
    class_label_feature = ClassLabel(num_classes=2, names=["True","False"])
    dataset = dataset.cast_column("label",class_label_feature)
    return dataset

def process_social_i_qa(dataset):
    dataset = dataset.class_encode_column("label")
    class_label_feature = ClassLabel(num_classes=3,names=["answerA","answerB","answerC"])
    dataset = dataset.cast_column("label",class_label_feature)
    return dataset

def process_commonsense_qa(dataset):
    dataset = dataset.rename_column("answerKey","label")
    dataset = dataset.class_encode_column("label")
    class_label_features = ClassLabel(num_classes=5,names=["0","1","2","3","4"])
    dataset = dataset.cast_column("label",class_label_features)

def process_cosmos_qa(dataset):
    dataset = dataset.class_encode_column("label")
    class_label_feature = ClassLabel(num_classes=4, names=['answer0', 'answer1', 'answer2','answer3'])
    dataset = dataset.cast_column('label', class_label_feature)
    return dataset

def process_hellaswag(dataset):
    dataset = dataset.class_encode_column("label")
    class_label_feature = ClassLabel(num_classes=4, names=['0', '1', '2','3'])
    # Cast the label column to ClassLabel
    dataset = dataset.cast_column('label', class_label_feature)

def process_mnli(dataset):
    dataset["validation"] = dataset.pop("validation_matched")
    return dataset

def load_hf_dataset(task_name:str,
                    debug:bool=False) -> DatasetDict:
    
    if task_name in DISK_TASKS:
        dataset = load_from_disk(DISK_TASKS[task_name])
        if task_name == "argument":
            dataset = dataset.rename_column(original_column_name="annotation",new_column_name="label")
            dataset = dataset.class_encode_column("label")
            num_labels = 3  
            class_label_feature = ClassLabel(num_classes=num_labels, names=['Argument_against', 'Argument_for', 'NoArgument'])
            dataset = dataset.cast_column('label', class_label_feature)
        
    elif task_name in GLUE_TASKS:
        dataset = load_dataset("glue",task_name)
    elif task_name in SUPER_GLUE_TASKS:
        dataset = load_dataset("super_glue",task_name)
    elif task_name == "scitail":
        dataset = load_dataset(task_name,"tsv_format")
        dataset = dataset.class_encode_column("label")
        class_label_feature = ClassLabel(num_classes=2, names=['entails', 'neutral'])
        # Cast the label column to ClassLabel
        dataset = dataset.cast_column('label', class_label_feature)
    elif task_name == "imdb":
        dataset = load_dataset(task_name)
        dataset["validation"] = dataset.pop("test")
    elif task_name == "winogrande":
        dataset = load_dataset(task_name,"winogrande_xl")
        dataset = dataset.rename_column("answer","label")
        dataset = dataset.class_encode_column("label")
        class_label_feature = ClassLabel(num_classes=2, names=['option1','option2'])
        dataset = dataset.cast_column('label', class_label_feature)
               
    else:
        dataset = load_dataset(task_name)
        if task_name == "boolq":
            dataset = dataset.rename_column("answer","label")
            #dataset = dataset.map(boolean_to_int_label)
            #dataset = dataset.cast_column("label",class_label_bool)
            dataset = dataset.class_encode_column("label")
            class_label_feature = ClassLabel(num_classes=2, names=["True","False"])
            dataset = dataset.cast_column("label",class_label_feature)

        if task_name == "social_i_qa":
            dataset = dataset.class_encode_column("label")
            class_label_feature = ClassLabel(num_classes=3,names=["answerA","answerB","answerC"])
            dataset = dataset.cast_column("label",class_label_feature)
        if task_name == "commonsense_qa":
            dataset = dataset.rename_column("answerKey","label")
            dataset = dataset.class_encode_column("label")
            class_label_features = ClassLabel(num_classes=5,names=["0","1","2","3","4"])
            dataset = dataset.cast_column("label",class_label_features)
        if task_name == "cosmos_qa":
            dataset = dataset.class_encode_column("label")
            class_label_feature = ClassLabel(num_classes=4, names=['answer0', 'answer1', 'answer2','answer3'])
            dataset = dataset.cast_column('label', class_label_feature)
        if task_name == "hellaswag":
            dataset = dataset.class_encode_column("label")
            class_label_feature = ClassLabel(num_classes=4, names=['0', '1', '2','3'])
            # Cast the label column to ClassLabel
            dataset = dataset.cast_column('label', class_label_feature)
        if task_name == "mnli":
            dataset["validation"] = dataset.pop("validation_matched")
            
    
    if debug:
        selected_datasets = {split: dataset[split].select(range(10)) for split in dataset.keys()}
        # Convert the dictionary back into a DatasetDict
        dataset = DatasetDict(selected_datasets)
    
    # TODO: Fix logger
    logging.info(f"Number of rows: {dataset.num_rows}")
    logging.info(f"Features for 'train' split: {dataset['train'].features}")
    logging.info(f"Sample from 'train' split: {dataset['train'][0]}")
    
    return dataset
    