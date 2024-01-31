from transformers import BertConfig
import adapters
from adapters import AutoAdapterModel


def setup_model(model_name,num_labels):
    
    config = BertConfig(
        model_name,num_labels=num_labels)
    
    model = AutoAdapterModel.from_pretrained(
        model_name,config=config)
    
    
def add_clf_adapter(task_name,model,config,num_labels):
    model.add_adapter(task_name,
                      config=adapters.BnConfig(
                          output_adapter=True,
                          mh_adapter=False,
                          reduction_factor=2,
                          non_linearity="relu"
    ))
    model.add_classification_head(
        task_name,
        num_labels=num_labels
    )
    model.train_adapter(task_name)
    
    #model.set_active_adapters(task_name)