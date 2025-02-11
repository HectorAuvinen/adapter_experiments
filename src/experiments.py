import os
import argparse
import logging
import json
import time
import shutil
from dataclasses import dataclass
from typing import Dict,List,Any,Union
from pathlib import Path

from transformers import set_seed
import torch
import adapters

from src.load_data import load_hf_dataset
from src.preprocessing import preprocess_dataset,get_tokenizer,get_encoding,get_label_count
from src.model_setup import setup_model,add_clf_adapter,add_mc_adapter,setup_ft_model_clf,setup_ft_model_mc
from src.training import TrainingParameters,get_trainer,get_ft_trainer
from src.file_utils import json_to_dict
from src.utils import get_key_by_value,get_max_len,get_seeds,get_tasks,logger_setup
from src.constants import MODEL_MAP,CLF_TASKS,MC_TASKS
from transformers import set_seed
from src.result_repository import ResultRepository

logger = logger_setup()


@dataclass
class ExperimentConfig:
    """
    Dataclass for storing experiment configurations.
    """
    task_name: str
    model_name: str
    output_path: str
    adapter_config: Dict[str, Any]
    training_config: Dict[str, Any]
    train_batch_size: int
    eval_batch_size: int
    max_len: str
    eval_column: str
    early_stopping: int
    keep_checkpoints: bool
    mode: str
    debug: bool
    seeds: List[int]

    @staticmethod
    def config_from_arguments(args: argparse.Namespace) -> "ExperimentConfig":
        """
        Parses arguments and instantiates the ExperimentConfig class.
        
        Args:
            args (argparse.Namespace): command-line arguments
        
        Returns:
            ExperimentConfig: an instance of the ExperimentConfig class
        """
        adapter_config = json_to_dict(args.adapter_config_path)
        
        training_config = json_to_dict(args.training_config_path)
        
        if args.single_config:
            logger.info("Using the first adapter configuration")
            first_config = next(iter(adapter_config))
            adapter_config = {first_key: adapter_config[first_key]}
        else:
            logger.info("Using all adapter configurations")

        
        seeds = [args.one_seed] if args.one_seed is not None else get_seeds(args.num_seeds)

        return ExperimentConfig(
            task_name=args.task_name,
            model_name=args.model_name,
            output_path=args.output_path,
            adapter_config=adapter_config,
            training_config=training_config,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            max_len=args.max_len,
            eval_column=args.eval_column,
            early_stopping=args.early_stopping,
            keep_checkpoints=args.keep_checkpoints,
            mode=args.mode,
            debug=args.debug,
            seeds=seeds)
        



class ExperimentRunner:
    """
    Class for running adapter and full fine-tune tune experiments. Takes an ExperimentConfig object and runs 
    adapter and/or full fine-tune experiments based on the config.
    
    Args:
        config (ExperimentConfig): configuration object for training parameters
    """
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.tasks = get_tasks(config.task_name)

    def run_experiments(self):
        """
        Run experiments based on run configuration.
        """
        for seed in self.config.seeds:
            logger.info(f"Running experiment with seed {seed}")
            if self.config.mode in ["adapter","all"]:
                self.run_adapter_experiment(seed)
            if self.config.mode in ["ft","all"]:
                self.run_ft_experiment(seed)
    
    def run_adapter_experiment(self, seed: int):
        """
        Trigger a single training & eval run for adapters

        Args:
            seed (str): seed to be used in running the experiment
        """
        logger.info("Starting adapter experiment with seed %d", seed)
        self.train_and_eval(
            tasks=self.tasks,
            model_name=self.config.model_name,
            output_dir=self.config.output_path,
            adapter_config=self.config.adapter_config,
            training_config=self.config.training_config,
            max_length=self.config.max_len,
            train_batch_size=self.config.train_batch_size,
            eval_column=self.config.eval_column,
            early_stopping=self.config.early_stopping,
            keep_checkpoints=self.config.keep_checkpoints,
            seed=seed,
            debug=self.config.debug)
    
    def run_ft_experiment(self, seed: int):
        """
        Trigger a single training & eval run for full fine-tuning
        
        Args:
            seed (str): seed to be used in running the experiment
        """

        logger.info("Starting fine-tuning experiment with seed %d", seed)
        self.ft_train_and_eval(
            tasks=self.tasks,
            model_name=self.config.model_name,
            output_dir=self.config.output_path,
            training_config=self.config.training_config,
            max_length=self.config.max_len,
            train_batch_size=self.config.train_batch_size,
            eval_column=self.config.eval_column,
            early_stopping=self.config.early_stopping,
            keep_checkpoints=self.config.keep_checkpoints,
            seed=seed,
            debug=self.config.debug
        )
        
    @staticmethod
    def ft_train_and_eval(tasks: List,
                          model_name: str,
                          output_dir: Union[Path,str],
                          training_config: Dict,
                          max_length: int,
                          train_batch_size: int,
                          eval_column: str,
                          early_stopping: int,
                          keep_checkpoints: bool,
                          seed: int,
                          debug: bool):
        """
        Training and evaluation pipeline for full fine-tuning. Given a set of tasks, a model is fine-tuned and evaluated, and the results are written 
        to the output directory.
        
        Args:
            tasks (list): set of tasks
            model_name (str): the name of the model to be trained, has to be in the model map
            output_dir (str): path to the directory for saving evaluations and model checkpoints
            training_config (dict): training configuration
            max_length (int): maximum length of sequence to be used
            train_batch_size (int): batch size to be used in training
            eval_column (int): name of entry for evaluation, to be specified if no "evaluation" entry in dataset
            early_stopping (int): number of epochs to wait for evaluation metric improvement before stopping training
            keep_checkpoints (bool): whether or not to save the model checkpoints
            seed (int): seed for reproducibility
            debug (bool): whether or not to subset the datasets for debugging
        """

        model_folder_name = MODEL_MAP[model_name]

        result_repo = ResultRepository(output_root=output_dir,model_key=model_name)
        
        config_dir = result_repo.create_config_dir(config_name="ft")
        seed_dir = result_repo.create_seed_dir(config_dir=config_dir,seed=seed)
        
        
        for task in tasks:
            logger.info(f"Running task: {task}")
            
            task_dir = result_repo.create_task_dir(seed_dir=seed_dir,task=task)
            
            max_length = get_max_len(max_length,task)
                
            output_eval_file = result_repo.get_output_eval_file(task_dir=task_dir,task=task)
            if output_eval_file.exists():
                logger.info(f"Evaluation already exists, skipping...")
                continue
                
            set_seed(seed)
            
            # load and prepare data
            data = load_hf_dataset(task,debug=debug)
            tokenizer = get_tokenizer(model_folder_name)
            encode = get_encoding(task)
            dataset = preprocess_dataset(data,encode,tokenizer,max_length)
            num_labels = get_label_count(dataset)
            
            # set up model (head with num labels)
            if task in CLF_TASKS:
                model = setup_ft_model_clf(model_folder_name,num_labels,dataset)
            elif task in MC_TASKS:
                model = setup_ft_model_mc(model_folder_name,num_labels,dataset)
            
            
            training_params = TrainingParameters(**training_config)
                
            train_args = training_params.get_training_arguments(training_params)
            
            # set up trainer
            trainer = get_ft_trainer(train_args,dataset,model,early_stopping=early_stopping,custom_eval=eval_column)

            start_time = time.time()
            trainer.train()
            end_time = time.time()
            training_time = end_time - start_time
            
            # evaluate,remove checkpoints and write results to file
            eval_results = trainer.evaluate()
                
            result_repo.write_eval_results(eval_results=eval_results,output_dir=str(task_dir),task=task,trainer=trainer,adapter_config=None,
                            batch_size=training_params.per_device_train_batch_size,max_length=max_length,training_time=training_time,early_stopping_patience=early_stopping)
            
            if not keep_checkpoints:
                result_repo.remove_directory(Path(task_dir,"checkpoints"))
        
    @staticmethod
    def train_and_eval(tasks: List,
                       model_name: str,
                       output_dir: Union[Path,str],
                       adapter_config: Dict,
                       training_config: Dict,
                       max_length: int,
                       train_batch_size: int,
                       eval_column: str,
                       early_stopping: int,
                       keep_checkpoints: bool,
                       seed: int,
                       debug: bool):
        """
        Training and evaluation pipeline for bottleneck adapter tuning. Given a set of tasks, a model is fine-tuned and evaluated, and the results are written 
        to output_dir.
        
        Args:
            tasks (list): set of tasks
            model_name (str): the name of the model to be trained, has to be in the model map
            output_dir (str): path to the directory for saving evaluations and model checkpoints
            adapter_config (dict): a set of adapter configurations to use
            training_config (dict): generic training parameters
            max_length (int): maximum length of sequence to be used
            train_batch_size (int): batch size to be used in training
            eval_column (int): name of entry for evaluation, to be specified if no "evaluation" entry in dataset
            early_stopping (int): number of epochs to wait for evaluation metric improvement before stopping training
            keep_checkpoints (bool): whether or not to save the model checkpoints
            seed (int): seed for reproducibility
            debug (bool): whether or not to subset the datasets for debugging
        """
        
        model_folder_name = MODEL_MAP[model_name]
        result_repo = ResultRepository(output_root=output_dir,model_key=model_name)
        
        for name,config in adapter_config.items():
            config_dir = result_repo.create_config_dir(config_name=name)
            seed_dir = result_repo.create_seed_dir(config_dir=config_dir,seed=seed)
        
            logger.info(f"running config: {config}")

            for task in tasks:
                logger.info(f"Running task: {task}")
                task_dir = result_repo.create_task_dir(seed_dir=seed_dir,task=task)
                
                max_length = get_max_len(max_length,task)
                
                output_eval_file = result_repo.get_output_eval_file(task_dir=task_dir,task=task)
                
                if output_eval_file.exists():
                   logger.info(f"Evaluation already exists, skipping...")
                   continue
               
                
                set_seed(seed)
                
                # prepare dataset
                data = load_hf_dataset(task,debug=debug)
                tokenizer = get_tokenizer(model_folder_name)
                encode = get_encoding(task)
                dataset = preprocess_dataset(data,encode,tokenizer,max_length)
                num_labels = get_label_count(dataset)

                # set up model and adapter
                model = setup_model(model_folder_name,num_labels,dataset)
                adapter_config = adapters.SeqBnConfig(**config)

                # add adapter
                if task in CLF_TASKS:
                    model = add_clf_adapter(task_name=task,model=model,num_labels=num_labels,adapter_config=adapter_config)
                elif task in MC_TASKS:
                    model = add_mc_adapter(task_name=task,model=model,num_labels=num_labels,adapter_config=adapter_config)
                else:
                    raise Exception("Task not defined in tasks")
                
                training_params = TrainingParameters(**training_config)
                    
                train_args = training_params.get_training_arguments(training_params)
                
                # set up trainer
                trainer = get_trainer(training_args=train_args,dataset=dataset,model=model,early_stopping=early_stopping,custom_eval=eval_column)

                # train & evaluate
                start_time = time.time()
                trainer.train()
                end_time = time.time()
                training_time = end_time - start_time
                
                eval_results = trainer.evaluate()
                
                    
                result_repo.write_eval_results(eval_results=eval_results,output_dir=str(task_dir),task=task,trainer=trainer,adapter_config=adapter_config,
                                batch_size=train_args.per_device_train_batch_size,max_length=max_length,training_time=training_time,early_stopping_patience=early_stopping)
                
                if not keep_checkpoints:
                    result_repo.remove_directory(Path(task_dir,"checkpoints"))