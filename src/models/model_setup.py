from transformers import BertConfig
import adapters
from adapters import AutoAdapterModel


def setup_model(model_name,num_labels):
    
    config = BertConfig.from_pretrained(
        model_name,num_labels=num_labels)
    
    model = AutoAdapterModel.from_pretrained(
        model_name,config=config)
    return model
    
    
def add_clf_adapter(task_name,model,num_labels,adapter_config):
    # TODO: ADD CONFIG INSTEAD OF HARD CODED
    model.add_adapter(task_name,
                      config=adapter_config)
    model.add_classification_head(
        task_name,
        num_labels=num_labels
    )
    model.train_adapter(task_name)
    
    return model
    
    #model.set_active_adapters(task_name)