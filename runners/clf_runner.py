# import argument parsers
import sys
import os
import json
import argparse
import logging
from pathlib import Path

#project_root = os.path.dirname(os.getcwd())
#sys.path.insert(0,project_root)
# Get the absolute path to the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the absolute path to the root of the project
project_root = os.path.dirname(script_dir)

# Add the project root to the sys.path
sys.path.insert(0, project_root)


from src.data_handling.load_data import *
from src.data_handling.preprocessing import *
from src.models.model_setup import *
from src.trainer.training import *
from src.trainer.file_utils import *
from transformers import set_seed

logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s : %(message)s')
logger = logging.getLogger(__name__)

root = Path(__file__).resolve().parent
ALL_TASKS = [
    "cb","rte","sick","mrpc","boolq","commonsense_qa",
    "argument","scitail","cosmos_qa","social_i_qa",
    "hellaswag","imdb","winogrande","sst2","qqp","mnli"]

CLF_TASKS = [
    "cb","rte","sick","mrpc","boolq",
    "argument","scitail","imdb","sst2","qqp","mnli"]   
    
    
MC_TASKS = [
    "commonsense_qa","cosmos_qa","social_i_qa",
    "hellaswag","winogrande"]

def train_and_eval(task,model,output_dir,adapter_config,training_config,seed):
    set_seed(seed)
    for name,config in adapter_config.items():
        logger.info(f"using config {name}")
        
        output_dir = os.path.join(output_dir,name)
        if not os.path.exists(output_dir):
            # If the folder doesn't exist, create it
            os.makedirs(output_dir)
            print(f"Folder '{output_dir}' created.")
        else:
            print(f"Folder '{output_dir}' already exists.")
        
            print(f"**********************************RUNNING CONFIG {name}*****************************")

        for task in tasks:
            print(f"**********************************RUNNING TASK {task}*****************************")
            # load dataset
            data = load_hf_dataset(task,debug=False)
            # get tokenizer (bert)
            tokenizer = get_tokenizer(model_name)
            # get encoding method for particular task
            encode = get_encoding(task)
            # apply encoding
            dataset = preprocess_dataset(data,encode,tokenizer)
            # get label count
            num_labels = get_label_count(dataset)
            # set up model (head with num labels)
            model = setup_model(model_name,num_labels,dataset)
            
            # set up adapter config
            ###########################3
            #adapter_config = adapters.SeqBnConfig(**config)
            adapter_config = adapters.SeqBnConfig(**config)
            print(f"Adapter config set up: {adapter_config}")
            #adapter_config = adapters.BnConfig(
            #                        output_adapter=config["output_adapter"],
            #                        mh_adapter=config["mh_adapter"],
            #                        reduction_factor=config["reduction_factor"],
            #                        non_linearity=config["non_linearity"])

            # add adapter
            model = add_clf_adapter(task_name=task,model=model,num_labels=num_labels,adapter_config=adapter_config)
            
            # set up training args
            final_output = os.path.join(output_dir,task)
            training_config["output_dir"] = final_output
            
            default_args = TrainingParameters(**training_config)
            print(f"Training arguments set up: {adapter_config}")
            #default_args = TrainingParameters(**training_args)
            #default_args = TrainingParameters(output_dir=final_output,
            #                                per_device_train_batch_size=8,
            #                                evaluation_strategy="epoch",
            #                                eval_steps=1,
            #                                save_strategy="epoch",
            #                                logging_steps=200)
            
            # TODO: FIX THIS BUG WHERE "linear" becomes ["linear"]
            default_args.lr_scheduler_type = "linear"
            train_args = get_training_arguments(default_args)
            
            # set up trainer
            trainer = get_trainer(train_args,dataset,model,early_stopping=3)
            # train
            trainer.train()
            
            # evaluate and write results to file
            eval_results = trainer.evaluate()
            print("results",eval_results)
            print("output_dir",output_dir)
            write_eval_results(eval_results,output_dir,task,trainer,adapter_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TODO")
    parser.add_argument("--task_name",type=str,help="TODO",default="cb")
    parser.add_argument("--model_name",type=str,help="TODO",default="bert-base-uncased")
    parser.add_argument("--output_path",type=str,help="TODO",default="outputs/evals")
    parser.add_argument("--adapter_config_path",type=str,help="TODO",default="src/configs/adapter_configs.json")
    parser.add_argument("--training_config_path",type=str,help="TODO",default="src/configs/training_config.json")
    parser.add_argument("--logging",type=str,default="INFO",help="log level")
    parser.add_argument("--multiple_adapters",action="store_true",help="TODO")
    parser.add_argument("--max_length",type=int,help="TODO",default=None)
    parser.add_argument("--train_batch_size",help="TODO",default=None)
    parser.add_argument("--eval_batch_size",help="TODO",default=None)
    #
    #"evaluation_strategy":"epoch",
    #"save_strategy":"epoch",
    #"learning_rate":1e-4,
    #"num_train_epochs":30,
    #"per_device_train_batch_size":8,
    #"per_device_eval_batch_size":8,
    #"eval_steps":1,
    #"logging_steps":200,
    #
    
    args = parser.parse_args()
    
    logging.getLogger().setLevel(level=args.logging)
    
    #outpath = Path(args.output_path)
    seed = 42
    tasks = args.task_name
    if tasks == "all":
        tasks = CLF_TASKS
    else:
        tasks = [args.task_name]
    model_name = args.model_name
    output_path = args.output_path
    adapter_config_path = args.adapter_config_path
    training_config_path = args.training_config_path
    
    adapter_config = json_to_dict(adapter_config_path)
    training_args = json_to_dict(training_config_path)
    if not args.multiple_adapters:
        logger.info("Using the first configuration")
        first_key = list(adapter_config.keys())[0]  # Get the first key
        adapter_config = {first_key: adapter_config[first_key]}
    else:
        logger.info("Using all configurations")
        
    train_and_eval(tasks,model_name,output_path,adapter_config,training_args,seed)