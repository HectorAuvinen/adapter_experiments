from transformers import BertConfig,AutoModelForSequenceClassification,AutoConfig,AutoModelForMultipleChoice
import adapters
from adapters import AutoAdapterModel


def setup_ft_model_clf(model_name,num_labels,dataset):
    id2label = {id: label for (id,label) in enumerate(dataset["train"].features["labels"].names)}
    #config = BertConfig.from_pretrained(
    config = AutoConfig.from_pretrained(
        model_name,id2label=id2label,num_labels=num_labels)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,config=config)
    return model

def setup_ft_model_mc(model_name,num_labels,dataset):
    id2label = {id: label for (id,label) in enumerate(dataset["train"].features["labels"].names)}
    #config = BertConfig.from_pretrained(
    config = AutoConfig.from_pretrained(
        model_name,id2label=id2label,num_labels=num_labels)
    
    model = AutoModelForMultipleChoice.from_pretrained(
        model_name,config=config)
    return model     

def setup_model(model_name,num_labels,dataset):
    id2label = {id: label for (id,label) in enumerate(dataset["train"].features["labels"].names)}
    config = BertConfig.from_pretrained(
    #config = AutoConfig.from_pretrained(
        model_name,id2label=id2label,num_labels=num_labels)
    
    model = AutoAdapterModel.from_pretrained(
        model_name,config=config)
    return model
    
    
def add_clf_adapter(task_name,model,num_labels,adapter_config):
    # TODO: ADD CONFIG INSTEAD OF HARD CODED
    # works for classification,NLI
    model.add_adapter(task_name,
                      config=adapter_config)
    model.add_classification_head(
        task_name,
        num_labels=num_labels
    )
    model.train_adapter(task_name)
    
    return model
    
    #model.set_active_adapters(task_name)

def add_mc_adapter(task_name,model,num_labels,adapter_config):
    model.add_adapter(task_name,config=adapter_config)
    
    model.add_multiple_choice_head(task_name,num_choices=num_labels)
    
    model.train_adapter(task_name)
    
    return model