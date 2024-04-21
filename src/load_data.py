import logging

from datasets import load_dataset,DatasetDict,ClassLabel,load_from_disk

from .constants import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="my_log_file.log" 
)


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
        dataset = process_argument(dataset)
    return dataset

def load_scitail(task_name,format="tsv_format"):
    """ Load and preprocess scitail """
    dataset = load_dataset(task_name,format)
    dataset = dataset.class_encode_column("label")
    class_label_feature = ClassLabel(num_classes=2, names=['entails', 'neutral'])
    dataset = dataset.cast_column('label', class_label_feature)
    return dataset

def load_winogrande(task_name,size="winogrande_xl"):
    """ Load and preprocess winogrande. By default the xl version is used """
    dataset = load_dataset(task_name,size)
    dataset = dataset.rename_column("answer","label")
    dataset = dataset.class_encode_column("label")
    class_label_feature = ClassLabel(num_classes=2, names=['option1','option2'])
    dataset = dataset.cast_column('label', class_label_feature)
    return dataset

def process_imdb(dataset):
    """ Preprocess IMDb by converting test to validation because this is what 
    the pipeline expects """
    dataset["validation"] = dataset.pop("test")
    return dataset

def process_boolq(dataset):
    """ Load and preprocess BoolQ """
    dataset = dataset.rename_column("answer","label")
    dataset = dataset.class_encode_column("label")
    class_label_feature = ClassLabel(num_classes=2, names=["True","False"])
    dataset = dataset.cast_column("label",class_label_feature)
    return dataset

def process_social_i_qa(dataset):
    """ Load and preprocess SocialIQA """
    dataset = dataset.class_encode_column("label")
    class_label_feature = ClassLabel(num_classes=3,names=["answerA","answerB","answerC"])
    dataset = dataset.cast_column("label",class_label_feature)
    return dataset

def process_commonsense_qa(dataset):
    """ Load and preprocess CSQA """
    dataset = dataset.rename_column("answerKey","label")
    dataset = dataset.class_encode_column("label")
    class_label_features = ClassLabel(num_classes=5,names=["0","1","2","3","4"])
    dataset = dataset.cast_column("label",class_label_features)
    return dataset

def process_cosmos_qa(dataset):
    """ Load and preprocess CosmosQA """
    dataset = dataset.class_encode_column("label")
    class_label_feature = ClassLabel(num_classes=4, names=['answer0', 'answer1', 'answer2','answer3'])
    dataset = dataset.cast_column('label', class_label_feature)
    return dataset

def process_hellaswag(dataset):
    """ Load and preprocess Hellaswag """
    dataset = dataset.class_encode_column("label")
    class_label_feature = ClassLabel(num_classes=4, names=['0', '1', '2','3'])
    dataset = dataset.cast_column('label', class_label_feature)
    return dataset

def process_mnli(dataset):
    """ Preprocess IMDb by converting validation_matched to validation because this is what 
    the pipeline expects """
    dataset["validation"] = dataset.pop("validation_matched")
    return dataset

def process_argument(dataset):
    """ Load and preprocess Argument """
    dataset = dataset.rename_column(original_column_name="annotation",new_column_name="label")
    dataset = dataset.class_encode_column("label")
    num_labels = 3  
    class_label_feature = ClassLabel(num_classes=num_labels, names=['Argument_against', 'Argument_for', 'NoArgument'])
    dataset = dataset.cast_column('label', class_label_feature)
    return dataset
    

def load_hf_dataset(task_name:str,
                    debug:bool=False) -> DatasetDict:
    """Function for loading a dataset from Hugging Face based on the task name. The
    16 supported datasets are specified in the README.

    Args:
        task_name (str): task name
        debug (bool, optional): whether to use debugging (only 10 first samples used). 
        Defaults to False.

    Returns:
        DatasetDict: The loaded and processed dataset ready for training and evaluation.
    """
    
    if task_name in DISK_TASKS:
        dataset = handle_disk_task(task_name)
        
    elif task_name in GLUE_TASKS:
        dataset = load_dataset("glue",task_name)
        if task_name == "mnli":
            dataset = process_mnli(dataset)
    elif task_name in SUPER_GLUE_TASKS:
        dataset = load_dataset("super_glue",task_name)
    elif task_name == "scitail":
        dataset = load_scitail(task_name=task_name)
    elif task_name == "winogrande":
        dataset = load_winogrande(task_name)        
    else:
        dataset = load_dataset(task_name)
        if task_name == "imdb":
            dataset = process_imdb(dataset)
        if task_name == "boolq":
            dataset = process_boolq(dataset)

        if task_name == "social_i_qa":
            dataset = process_social_i_qa(dataset)
        if task_name == "commonsense_qa":
            dataset = process_commonsense_qa(dataset)
        if task_name == "cosmos_qa":
            dataset = process_cosmos_qa(dataset)
        if task_name == "hellaswag":
            dataset = process_hellaswag(dataset)
        if task_name == "mnli":
            dataset = process_mnli(dataset)
            
    
    if debug:
        # only use 10 samples
        selected_datasets = {split: dataset[split].select(range(10)) for split in dataset.keys()}
        dataset = DatasetDict(selected_datasets)
    
    return dataset
    