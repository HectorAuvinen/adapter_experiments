from transformers import AutoModelForSequenceClassification,AutoConfig,AutoModelForMultipleChoice
import adapters
from adapters import AutoAdapterModel


def setup_ft_model_clf(model_name,num_labels,dataset):
    """
    Initializes and configures a model for full fine-tune sequence classification with a pretrained model.

    Args:
        model_name (str): The name of the pretrained model to be used. Should be compatible with models available in the Hugging Face model hub.
        num_labels (int): The number of unique labels in the classification task.
        dataset (DatasetDict): The dataset to be used.

    Returns:
        AutoModelForSequenceClassification: configured model ready for training.
    """
    id2label = {id: label for (id,label) in enumerate(dataset["train"].features["labels"].names)}
    config = AutoConfig.from_pretrained(
        model_name,id2label=id2label,num_labels=num_labels)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,config=config)
    return model

def setup_ft_model_mc(model_name,num_labels,dataset):
    """
    Initializes and configures a model for full fine-tune multiple choice answering with a pretrained model.

    Args:
        model_name (str): The name of the pretrained model to be used. Should be compatible with models available in the Hugging Face model hub.
        num_labels (int): The number of unique labels in the classification task.
        dataset (DatasetDict): The dataset to be used.

    Returns:
        AutoModelForMultipleChoice: configured model ready for training.
    """
    
    id2label = {id: label for (id,label) in enumerate(dataset["train"].features["labels"].names)}
    config = AutoConfig.from_pretrained(
        model_name,id2label=id2label,num_labels=num_labels)
    
    model = AutoModelForMultipleChoice.from_pretrained(
        model_name,config=config)
    return model     

def setup_model(model_name,num_labels,dataset):
    """
    Initializes and configures a model for adapter training with a pretrained model.

    Args:
        model_name (str): The name of the pretrained model to be used. Should be compatible with models available in the Hugging Face model hub.
        num_labels (int): The number of unique labels in the classification task.
        dataset (DatasetDict): The dataset to be used.

    Returns:
        AutoAdapterModel: configured model ready for adapter training.
    """
    id2label = {id: label for (id,label) in enumerate(dataset["train"].features["labels"].names)}
    config = AutoConfig.from_pretrained(
        model_name,id2label=id2label,num_labels=num_labels)
    
    model = AutoAdapterModel.from_pretrained(
        model_name,config=config)
    return model
    
    
def add_clf_adapter(task_name,model,num_labels,adapter_config):
    """
    Adds a classification head and adapter to a model.
    
    Args:
        task_name (str): The name of the task to save the adapter.
        model (AutoAdapterModel): The model to which the adapter will be added.
        num_labels (int): The number of labels for the task.
        adapter_config (BnConfig): Configuration settings for the adapter.

    Returns:
        AutoAdapterModel: Model with a classification head and adapter.

    """
    model.add_adapter(task_name,
                      config=adapter_config)
    model.add_classification_head(
        task_name,
        num_labels=num_labels
    )
    model.train_adapter(task_name)
    
    return model

def add_mc_adapter(task_name,model,num_labels,adapter_config):
    """
    Adds a multiple choice head and adapter to a model.
    
    Args:
        task_name (str): The name of the task to save the adapter.
        model (AutoAdapterModel): The model to which the adapter will be added.
        num_labels (int): The number of labels for the task.
        adapter_config (BnConfig): Configuration settings for the adapter.

    Returns:
        AutoAdapterModel: Model with a classification head and adapter.

    """
    model.add_adapter(task_name,config=adapter_config)
    
    model.add_multiple_choice_head(task_name,num_choices=num_labels)
    
    model.train_adapter(task_name)
    
    return model