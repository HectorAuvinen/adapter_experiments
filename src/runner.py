import sys
import os
import argparse
import logging
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
root = Path(__file__).resolve().parent

from src.experiments import ExperimentConfig,ExperimentRunner

if __name__ == '__main__':
    """
    Runner for conducting experiments on the supported datasets.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name",type=str,help="Task or tasks to conduct the experiments on. See README for supported tasks",default="argument")
    parser.add_argument("--model_name",type=str,help="Model to use in the experiments. See README for supported models",default="bert-tiny-uncased")
    parser.add_argument("--output_path",type=str,help="Output path for the experiment results (model performances and configurations). \
                        If you want to keep the trained models, use the keep_checkpoints flag",default="outputs/evals")
    parser.add_argument("--adapter_config_path",type=str,help="Path to the adapter configuration path. Expects a dictionary where keys \
                        are config names and values are configuration dictionaries. See src/configs/adapter_configs.json for an example",
                        default="src/configs/adapter_config.json")
    parser.add_argument("--training_config_path",type=str,help="Path to the Transformers training config. See src/configs/training_config.json for an example",
                        default="src/configs/training_config.json")
    parser.add_argument("--single_config",action="store_true",help="Whether to use the first config entry of a dictionary of multiple configs (debugging)")
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
    
    args = parser.parse_args()

    config = ExperimentConfig.config_from_arguments(args)

    runner = ExperimentRunner(config=config)
    runner.run_experiments()