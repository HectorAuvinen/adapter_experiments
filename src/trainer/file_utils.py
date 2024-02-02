# import
import os

def write_eval_results(eval_results,output_dir,task,trainer):
    print("Writing eval results")
    print(eval_results)
    output_eval_file = os.path.join(output_dir,f"eval_results_{task}.txt")
    if trainer.is_world_process_zero():
        with open(output_eval_file, 'w') as writer:
            for key,value in eval_results.items():
                writer.write("%s = %s\n" % (key,value))
    
    