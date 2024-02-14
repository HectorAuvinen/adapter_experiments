# import
import os
import json

def json_to_dict(file_path):
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
    return data

def write_eval_results(eval_results,output_dir,task,trainer,adapter_config,batch_size,max_length,training_time):
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
            writer.write("training time (seconds) = %.2f\n" % training_time)
            for config_key,config_value in config.items():
                writer.write("%s = %s\n" % (config_key,config_value))
            for key,value in eval_results.items():
                writer.write("%s = %s\n" % (key,value))
    
    