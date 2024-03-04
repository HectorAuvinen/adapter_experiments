# import
import os
import json
from pathlib import Path

def json_to_dict(file_path):
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
    return data

def write_eval_results(eval_results,output_dir,task,trainer,adapter_config,batch_size,max_length,training_time,early_stopping_patience):
    print("Writing eval results")
    print(eval_results)
    config = {}
    for key in adapter_config:
        config[key] = adapter_config[key]
    #for attr_name, attr_value in vars(adapter_config).items():
    #    print(f"{attr_name}: {attr_value}")
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
    
    
    
def read_eval_results(path):
    res_path = Path(path)
    trainingtime = 0
    new_results = {}
    for config in res_path.iterdir():
        config_results = {config.name:{}}
        for seed in config.iterdir():
            config_results[config.name][seed.name] = {}
            for dataset in seed.iterdir():
                task = dataset.name.split("eval_results_")[-1].split(".txt")[0]
                try:
                    with open(dataset,"r") as file:
                        lines = file.readlines()
                        time = [float(detail.split("=")[1].strip()) for detail in lines if "training time" in detail][0]
                        trainingtime += time
                        line = [float(line.strip().split("= ")[-1]) for line in lines if "eval_accuracy" in line][0]
                        config_results[config.name][seed.name][task] = line
                except PermissionError as e:
                    continue
        new_results.update(config_results)
    return new_results