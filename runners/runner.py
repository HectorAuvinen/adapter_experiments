# import argument parsers
import sys
import os
import json
import argparse
import logging
import time
import shutil
from pathlib import Path


from transformers import set_seed
import torch

#project_root = os.path.dirname(os.getcwd())
#sys.path.insert(0,project_root)
# Get the absolute path to the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the absolute path to the root of the project
project_root = os.path.dirname(script_dir)

# Add the project root to the sys.path
sys.path.insert(0, project_root)


from src.load_data import *
from src.preprocessing import *
from src.model_setup import *
from src.training import *
from src.file_utils import write_eval_results,json_to_dict
from src.utils import get_seeds,get_key_by_value,get_tasks,get_max_len
# from src.constants import
from src.constants import *
from transformers import set_seed

logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s : %(message)s')
logger = logging.getLogger(__name__)

root = Path(__file__).resolve().parent


def ft_train_and_eval(task,model,output_dir,training_config,max_length,train_batch_size,eval_column,early_stopping,keep_checkpoints,seed,debug):
    model_folder_name = get_key_by_value(model,MODEL_MAP)
    result_root = Path(output_dir)/Path(model_folder_name)
    config_dir = os.path.join(result_root,"full_fine_tune")
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
        print(f"Folder '{config_dir}' created.")
    else:
        print(f"Folder '{config_dir}' already exists.")
    
    output_dir = os.path.join(config_dir,str(seed))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Folder '{output_dir}' created.")
    else:
        print(f"Folder '{output_dir}' already exists.")
    
    for task in tasks:
        print(f"**********************************RUNNING TASK {task}*****************************")
        """        
        if max_length == "max":
            max_length = None
        elif max_length == "std":
            max_length = MAX_LENS[task]
        else:
            max_length = int(max_length)
        """
        max_length = get_max_len(max_length,task)
            
        output_eval_file = os.path.join(output_dir,f"eval_results_{task}.txt")
        if os.path.isfile(output_eval_file):
            print("EVAL EXISTS, SKIPPING")
            continue
        
        # set seed
        set_seed(seed)
        # load dataset
        data = load_hf_dataset(task,debug=debug)
        # get tokenizer (bert)
        tokenizer = get_tokenizer(model_name)
        # get encoding method for particular task
        encode = get_encoding(task)
        # apply encoding
        dataset = preprocess_dataset(data,encode,tokenizer,max_length)
        # get label count
        num_labels = get_label_count(dataset)
        # set up model (head with num labels)
        if task in CLF_TASKS:
            model = setup_ft_model_clf(model_name,num_labels,dataset)
        elif task in MC_TASKS:
            model = setup_ft_model_mc(model_name,num_labels,dataset)
        
        
        # set up training args
        final_output = os.path.join(output_dir,task)
        training_config["output_dir"] = final_output
        
        default_args = TrainingParameters(**training_config)
            
        
        default_args.lr_scheduler_type = "linear"
        print("TRAIN BATCH SIZE:",train_batch_size)
        if train_batch_size:
            print(f"Changing batchs size from {default_args.per_device_train_batch_size} to {train_batch_size}")
            default_args.per_device_train_batch_size = train_batch_size
        train_args = get_training_arguments(default_args)
        
        # set up trainer
        trainer = get_ft_trainer(train_args,dataset,model,early_stopping=early_stopping,custom_eval=eval_column)

        # train
        start_time = time.time()
        trainer.train()
        end_time = time.time()
        
        training_time = end_time - start_time
        
        # evaluate,remove checkpoints and write results to file
        eval_results = trainer.evaluate()
        if not keep_checkpoints:
            checkpoint_dir = Path(final_output)
            shutil.rmtree(checkpoint_dir)
            try:
                shutil.rmtree(checkpoint_dir)
                print(f"Successfully removed directory: {checkpoint_dir}")
            except Exception as e:
                print(f"Error removing directory {checkpoint_dir}: {e}")
            
        print("results",eval_results)
        print("output_dir",output_dir)
        write_eval_results(eval_results,output_dir,task,trainer,adapter_config,
                        default_args.per_device_train_batch_size,max_length,training_time,early_stopping)
        del model
        torch.cuda.empty_cache()
    


def train_and_eval(task,model,output_dir,adapter_config,training_config,max_length,train_batch_size,eval_column,early_stopping,keep_checkpoints,seed,debug):
    # the parent folder for all experiments: e.g. target_folder/bert-base-uncased
    model_folder_name = get_key_by_value(model,MODEL_MAP)
    result_root = Path(output_dir)/Path(model_folder_name)
    # the task specific folder in the experiments parent folder: e.g. target_folder/bert-base-uncased/cb
    #dataset_results = Path(result_root)/Path(task.split("/")[-1])
        
    for name,config in adapter_config.items():
        logger.info(f"using config {name}")
        
        # the configuration specific folder in the experiments parent folder: e.g. target_folder/bert-base-uncased/redf_16
        config_dir = os.path.join(result_root,name)
        if not os.path.exists(config_dir):
            # If the folder doesn't exist, create it
            os.makedirs(config_dir)
            print(f"Folder '{config_dir}' created.")
        else:
            print(f"Folder '{config_dir}' already exists.")
        
        output_dir = os.path.join(config_dir,str(seed))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Folder '{output_dir}' created.")
        else:
            print(f"Folder '{output_dir}' already exists.")
            
        print(f"**********************************RUNNING CONFIG {name}*****************************")

        for task in tasks:
            print(f"**********************************RUNNING TASK {task}*****************************")
            """            if max_length == "max":
                max_length = None
            elif max_length == "std":
                max_length = MAX_LENS[task]
            else:
                max_length = int(max_length)
            """
                
            max_length = get_max_len(max_length,task)
            
            output_eval_file = os.path.join(output_dir,f"eval_results_{task}.txt")
            if os.path.isfile(output_eval_file):
                print("EVAL EXISTS, SKIPPING")
                continue
                
            # set seed
            set_seed(seed)
            # load dataset
            data = load_hf_dataset(task,debug=debug)
            # get tokenizer (bert)
            tokenizer = get_tokenizer(model_name)
            # get encoding method for particular task
            encode = get_encoding(task)
            # apply encoding
            dataset = preprocess_dataset(data,encode,tokenizer,max_length)
            # get label count
            num_labels = get_label_count(dataset)
            # set up model (head with num labels)
            model = setup_model(model_name,num_labels,dataset)
            
            # set up adapter config
            ###########################3
            adapter_config = adapters.SeqBnConfig(**config)
            # print(f"Adapter config set up: {adapter_config}")
            #adapter_config = adapters.BnConfig(
            #                        output_adapter=config["output_adapter"],
            #                        mh_adapter=config["mh_adapter"],
            #                        reduction_factor=config["reduction_factor"],
            #                        non_linearity=config["non_linearity"])

            # add adapter
            if task in CLF_TASKS:
                print("Adding classification adapter")
                model = add_clf_adapter(task_name=task,model=model,num_labels=num_labels,adapter_config=adapter_config)
            elif task in MC_TASKS:
                print("Adding multiple choice adapter")
                model = add_mc_adapter(task_name=task,model=model,num_labels=num_labels,adapter_config=adapter_config)
            else:
                raise Exception("Task not defined in tasks")
            # set up training args
            final_output = os.path.join(output_dir,task)
            training_config["output_dir"] = final_output
            
            default_args = TrainingParameters(**training_config)
                
            # print(f"Training arguments set up: {adapter_config}")
            #default_args = TrainingParameters(**training_args)
            #default_args = TrainingParameters(output_dir=final_output,
            #                                per_device_train_batch_size=8,
            #                                evaluation_strategy="epoch",
            #                                eval_steps=1,
            #                                save_strategy="epoch",
            #                                logging_steps=200)
            
            # TODO: FIX THIS BUG WHERE "linear" becomes ["linear"]
            default_args.lr_scheduler_type = "linear"
            print("TRAIN BATCH SIZE:",train_batch_size)
            if train_batch_size:
                print("###########################################################################################")
                print(f"Changing batchs size from {default_args.per_device_train_batch_size} to {train_batch_size}")
                default_args.per_device_train_batch_size = train_batch_size
            train_args = get_training_arguments(default_args)
            
            # set up trainer
            trainer = get_trainer(train_args,dataset,model,early_stopping=early_stopping,custom_eval=eval_column)

            # train
            start_time = time.time()
            trainer.train()
            end_time = time.time()
            
            training_time = end_time - start_time
            
            # evaluate and write results to file
            eval_results = trainer.evaluate()
            if not keep_checkpoints:
                checkpoint_dir = Path(final_output)
                shutil.rmtree(checkpoint_dir)
                try:
                    shutil.rmtree(checkpoint_dir)
                    print(f"Successfully removed directory: {checkpoint_dir}")
                except Exception as e:
                    print(f"Error removing directory {checkpoint_dir}: {e}")
                
            print("results",eval_results)
            print("output_dir",output_dir)
            write_eval_results(eval_results,output_dir,task,trainer,adapter_config,
                            default_args.per_device_train_batch_size,max_length,training_time,early_stopping)
            del model
            torch.cuda.empty_cache()


if __name__ == '__main__':
    """
    Runner for conducting experiments on the supported datasets.
    """
    parser = argparse.ArgumentParser(description="TODO")
    parser.add_argument("--task_name",type=str,help="Task or tasks to conduct the experiments on. See README for supported tasks",default="argument")
    parser.add_argument("--model_name",type=str,help="Model to use in the experiments. See README for supported models",default="bert-tiny-uncased")
    parser.add_argument("--output_path",type=str,help="Output path for the experiment results (model performances and configurations). \
                        If you want to keep the trained models, use the keep_checkpoints flag",default="outputs/evals")
    parser.add_argument("--adapter_config_path",type=str,help="Path to the adapter configuration path. Expects a dictionary where keys \
                        are config names and values are configuration dictionaries. See src/configs/adapter_configs.json for an example",
                        default="src/configs/adapter_config.json")
    parser.add_argument("--training_config_path",type=str,help="Path to the Transformers training config. See src/configs/training_config.json for an example",
                        default="src/configs/training_config.json")
    parser.add_argument("--logging",type=str,default="DEBUG",help="log level")
    parser.add_argument("--single_config",action="store_true",help="Whether to use the first config entry of a dictionary of multiple configs (debugging)")
    # parser.add_argument("--max_length",type=int,help="Maximum sequence length to use",default=None)
    parser.add_argument("--train_batch_size",help="Training batch size. Can also be configured in the training config.",default=None)
    parser.add_argument("--eval_batch_size",help="Evaluation batch size. Can also be configured in the training config.",default=None)
    parser.add_argument("--max_len",type=str,help="Maximum sequence length to use. std maps to predefined max lengths (256 for classification, 128 for multiple choice), \
                        max maps to None (no maximum length), and any int maps to maximum length of that int",default="std")
    parser.add_argument("--eval_column",type=str,help="Custom column to use in evaluation. By default the validation column is used but if you \
                        want to use a different column define the column name here",default=None)
    parser.add_argument("--num_seeds",type=int,help="Number of seeds to use for the experiments. Each configuration is run num_seeds times and saved \
                        into a corresponding seed folder in output_path",default=3)
    parser.add_argument("--early_stopping",type=int,help="Tolerance for early stopping",default=3)
    parser.add_argument("--keep_checkpoints",help="Whether to save the training checkpoints. False by default",action="store_true")
    parser.add_argument("--mode",choices=["adapter","ft","all"],help="Whether to train adapters, do full fine-tuning or both. Defaults to adapters",default="adapter")
    parser.add_argument("--debug",action="store_true",help="Debugging mode. Datasets are sliced to contain only 10 samples.")
    parser.add_argument("--one_seed",type=int,default=None)
    
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
    # seed = [14,42,808]
    if args.one_seed:
        seeds = [args.one_seed]
    else:
        seeds = get_seeds(args.num_seeds)
    
    """
    tasks = args.task_name
        if tasks == "all":
        tasks = ALL_TASKS
    elif tasks == "subset":
        tasks = SUBSET_TASKS
    elif tasks == "subset_2":
        tasks = SUBSET_TASKS_2
    elif tasks == "subset_3":
        tasks = SUBSET_TASKS_3
    elif tasks == "subset_4":
        tasks = SUBSET_TASKS_4
    elif tasks == "clf":
        tasks = CLF_TASKS
    elif tasks == "mc":
        tasks = MC_TASKS
    else:
        tasks = [args.task_name]"""
    tasks = get_tasks(args.task_name)
    #model_name = args.model_name
    output_path = args.output_path
    adapter_config_path = args.adapter_config_path
    training_config_path = args.training_config_path
    train_batch_size = args.train_batch_size if not args.train_batch_size else int(args.train_batch_size)
    early_stopping = args.early_stopping
    eval_column = args.eval_column
    keep_checkpoints = args.keep_checkpoints
    mode = args.mode
    debug = args.debug
    
    adapter_config = json_to_dict(adapter_config_path)
    training_args = json_to_dict(training_config_path)
    
    if args.single_config:
        logger.info("Using the first configuration")
        first_key = list(adapter_config.keys())[0]
        adapter_config = {first_key: adapter_config[first_key]}
    else:
        logger.info("Using all configurations")
    
    max_len = args.max_len
    """    
    if max_len == "max":
        max_len = None
    elif max_len == "std":
        max_len = MAX_LENS[args.task_name]
    else:
        max_len = int(max_len)
    """
    #
    model_name = MODEL_MAP[args.model_name]
    
    print("MAX LEN",max_len)
    train_start = time.time()
    for seed in seeds:
        if mode == "adapter" or mode == "all":
            train_and_eval(tasks,model_name,output_path,adapter_config,training_args,
                        max_len,train_batch_size,eval_column,early_stopping,keep_checkpoints,
                        seed,debug)
        if mode == "ft" or mode == "all":
            ft_train_and_eval(task=tasks,model=model_name,output_dir=output_path,
                              training_config=training_args,max_length=max_len,
                              train_batch_size=train_batch_size,eval_column=eval_column,early_stopping=early_stopping,
                              keep_checkpoints=keep_checkpoints,seed=seed,debug=debug)
            # 
            # task,model,output_dir,training_config,max_length,train_batch_size,eval_column,early_stopping,keep_checkpoints,seed)
    
    train_end = time.time()
    total_time = train_end - train_start
    print("TOTAL TIME TRAINED",total_time)