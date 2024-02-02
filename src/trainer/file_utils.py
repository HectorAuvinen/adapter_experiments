# import
import os

def write_eval_results(eval_results,output_dir,task,trainer,adapter_config):
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
            writer.write(str(config))
            for key,value in eval_results.items():
                writer.write("%s = %s\n" % (key,value))
    
    