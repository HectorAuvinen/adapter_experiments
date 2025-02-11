import logging
import os
from pathlib import Path

from src.utils import logger_setup

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)

class ResultRepository:
    """
    A repository for handling output resources from the experiments.
    
    """
    def __init__(self, output_root: str, model_key: str):
        self.output_root = Path(output_root)
        self.model_key = model_key

    def get_model_dir(self) -> Path:
        """
        Create a directory for the model if necessary and return the path.
        """
        model_dir = self.output_root / self.model_key
        if not model_dir.exists():
            model_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Created model directory: %s", model_dir)
        return model_dir

    def create_config_dir(self, config_name: str = "full_fine_tune") -> Path:
        """
        Create a directory for the configuration if necessary and return the path.
        
        Args:
            config_name (str): name of the configuration.
        """
        config_dir = self.get_model_dir() / config_name
        if not config_dir.exists():
            config_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Created configuration directory: %s", config_dir)
        return config_dir

    def create_seed_dir(self, config_dir: Path, seed: int) -> Path:
        """
        Create a directory in the configuration path for a seed if necessary and return the path.
        
        Args:
            config_dir (Path): path to the configuration directory
            seed (int): current seed
        """
        seed_dir = config_dir / str(seed)
        if not seed_dir.exists():
            seed_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Created seed directory: %s", seed_dir)
        return seed_dir

    def create_task_dir(self, seed_dir: Path, task: str) -> Path:
        """
        Create a directory in the seed path for a task if necessary and return the path.
        
        Args:
            seed (Path): path to the seed directory
            task (str): current task
        """
        task_dir = seed_dir / task
        if not task_dir.exists():
            task_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Created task directory: %s", task_dir)
        return task_dir
    
    @staticmethod
    def get_output_eval_file(task_dir: Path,task:str) -> Path:
        """
        Get the evaluation file path
        
        Args:
            task_dir (Path): path to the task directory
            task (str): current task
        """
        
        output_eval_file = task_dir / f"eval_results_{task}.txt"
        return output_eval_file
            
        

    def remove_directory(self, dir_path: Path):
        """
        Remove a given directory
        
        Args:
            dir_path (Path): path to the directory
        """
        try:
            import shutil
            shutil.rmtree(dir_path)
            logger.info("Successfully removed directory: %s", dir_path)
        except Exception as e:
            logger.error("Error removing directory %s: %s", dir_path, e)
            
    @staticmethod
    def write_eval_results(eval_results,output_dir,task,trainer,adapter_config,batch_size,max_length,training_time,early_stopping_patience):
        """ 
        Write the experiment (hyper)parameters and performance results into a file
        """
        config = {}
        if adapter_config:
            for key in adapter_config:
                config[key] = adapter_config[key]
        output_eval_file = os.path.join(output_dir,f"eval_results_{task}.txt")
        
        with open(output_eval_file, 'w') as writer:
            writer.write("batch size = %s\n" % batch_size)
            writer.write("max length = %s\n" % max_length)
            writer.write("early stopping patience = %s\n" % early_stopping_patience)
            writer.write("training time (seconds) = %.2f\n" % training_time)
            for config_key,config_value in config.items():
                writer.write("%s = %s\n" % (config_key,config_value))
            for key,value in eval_results.items():
                writer.write("%s = %s\n" % (key,value))