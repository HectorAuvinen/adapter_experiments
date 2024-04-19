import os
import json
from pathlib import Path

import numpy as np

from .constants import SUBSET_TASKS_4

def json_to_dict(file_path):
    """ Read a JSON into a dictionary"""
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
    return data

def write_eval_results(eval_results,output_dir,task,trainer,adapter_config,batch_size,max_length,training_time,early_stopping_patience):
    """ Write the experiment (hyper)parameters and performance results into a file"""
    config = {}
    for key in adapter_config:
        config[key] = adapter_config[key]
    output_eval_file = os.path.join(output_dir,f"eval_results_{task}.txt")
    if trainer.is_world_process_zero():
        with open(output_eval_file, 'w') as writer:
            writer.write("batch size = %s\n" % batch_size)
            writer.write("max length = %s\n" % max_length)
            writer.write("early stopping patience = %s\n" % early_stopping_patience)
            writer.write("training time (seconds) = %.2f\n" % training_time)
            for config_key,config_value in config.items():
                writer.write("%s = %s\n" % (config_key,config_value))
            for key,value in eval_results.items():
                writer.write("%s = %s\n" % (key,value))
    
    
    
def read_eval_results(path,two_datasets=False,skip=None,show_batch_and_len=False):
    """Read the evaluation results from a given path"""
    res_path = Path(path)
    trainingtime = 0
    new_results = {}
    batch_sizes = {}
    max_lengths = {}
    for config in res_path.iterdir():
        if config.name == skip:
            continue
        config_results = {config.name:{}}
        for seed in config.iterdir():
            config_results[config.name][seed.name] = {}
            for dataset in seed.iterdir():
                task = dataset.name.split("eval_results_")[-1].split(".txt")[0]
                if two_datasets:
                    # only consider last experiment setup with sst2 and sick
                    if task not in SUBSET_TASKS_4:
                        continue
                try:
                    with open(dataset,"r") as file:
                        lines = file.readlines()

                        current_batch_size = int([line.split("=")[1].strip() for line in lines if "batch size" in line][0])
                        current_max_length = int([line.split("=")[1].strip() for line in lines if "max length" in line][0])
                        
                        if task in batch_sizes and max_lengths:
                            batch_sizes[task].append(current_batch_size)
                            max_lengths[task].append(current_max_length)
                        else:
                            batch_sizes[task] = [current_batch_size]
                            max_lengths[task] = [current_max_length]

                        time = [float(detail.split("=")[1].strip()) for detail in lines if "training time" in detail][0]
                        trainingtime += time
                        accuracy = [float(line.strip().split("= ")[-1]) for line in lines if "eval_accuracy" in line][0]
                        config_results[config.name][seed.name][task] = accuracy
                except PermissionError as e:
                    continue
        new_results.update(config_results)
    if show_batch_and_len:
        print("batches",batch_sizes)
        print("lengths",max_lengths)
    return new_results

def read_eval_results_2(root_path,to_skip=None,two_datasets=False):
    """Read the evaluation results from a given path (format for statistical tests and line plots)"""
    results = {}
    for model_folder in root_path.iterdir():
        if model_folder.is_dir():
            model_name = model_folder.name
            if to_skip and model_name == to_skip:
                continue
            for config_folder in model_folder.iterdir():
                reduction_factor = 'full_fine_tune' if 'full_fine_tune' in config_folder.name else config_folder.name
                for seed_folder in config_folder.iterdir():
                    for eval_file in seed_folder.glob("eval_results_*.txt"):
                        with eval_file.open() as f:
                            content = f.read()
                            dataset_name = eval_file.stem.split('eval_results_')[-1]
                            if two_datasets and dataset_name not in ['sick', 'sst2']:
                                continue
                            accuracy = float([line.split('=')[1] for line in content.splitlines() if 'eval_accuracy' in line][0])
                            reduction_factor_value = "fft" if reduction_factor == 'full_fine_tune' else float(
                                [line.split('=')[1] for line in content.splitlines() if 'reduction_factor' in line][0])

                            if dataset_name not in results:
                                results[dataset_name] = {}
                            if model_name not in results[dataset_name]:
                                results[dataset_name][model_name] = {}
                            if reduction_factor_value not in results[dataset_name][model_name]:
                                results[dataset_name][model_name][reduction_factor_value] = []

                            results[dataset_name][model_name][reduction_factor_value].append(accuracy)
                            
    for dataset, models in results.items():
        for model, reduction_factors in models.items():
            for rf, accuracies in reduction_factors.items():
                mean_accuracy = np.mean(accuracies)
                std_dev = np.std(accuracies)
                results[dataset][model][rf] = {'mean_accuracy': mean_accuracy, 'std_dev': std_dev}

    return results





## old format utils below ##

def get_dataset_and_acc(file_path,name_map):
    """ Read eval results from path (oldest output format)"""
    name = Path(file_path).name.split(".txt")[0].split("results_")[-1]
    contents = []
    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            if "}" in line:
                a,b = line.split("}")
                contents.append(a)
                contents.append(b)
            else:
                contents.append(line)
    accuracy_lines = [line.rstrip('\n').strip() for line in contents if 'eval_accuracy' in line]
    new_name = name_map[name]
    return new_name,float(accuracy_lines[0].split("= ")[-1])*100

def txt_to_dict(res_path,name_map):
    """ go over results in a path and collect results in a dictionary (oldest output format)"""
    new_results = {}
    for file in res_path.iterdir():
        if file.is_file() and "eval_results" in str(file):
            name,acc = get_dataset_and_acc(file,name_map)
            new_results[name] = acc

    return new_results


def get_dataset_and_acc_2(file_path,name_map):
    """ Read eval results from path (old output format)"""
    name = Path(file_path).name.split(".txt")[0].split("results_")[-1]
    contents = []
    with open(file_path, "r") as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines if "eval_accuracy" in line]
        acc = float(lines[0].split("= ")[-1])*100
        new_name = name_map[name]
        return new_name,acc
        
def text_to_dict_2(res_path,name_map):
    """ go over results in a path and collect results in a dictionary (old output format)"""
    new_results = {}
    for file in res_path.iterdir():
        if file.is_file() and "eval_results" in str(file.name):
            name,acc = get_dataset_and_acc_2(file,name_map)
            new_results[name] = acc

    return new_results