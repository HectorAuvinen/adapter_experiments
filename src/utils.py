import logging

from .constants import PREDEFINED_SEEDS,ALL_TASKS,SUBSET_TASKS,SUBSET_TASKS_2,SUBSET_TASKS_3,SUBSET_TASKS_4,CLF_TASKS,MC_TASKS,MAX_LENS

def get_seeds(n_seeds):
    """
    Returns a consistent list of seeds based on number of seeds.
    
    """

    if n_seeds > len(PREDEFINED_SEEDS):
        raise ValueError(f"Requested number of seeds exceeds the limit of {len(PREDEFINED_SEEDS)}")
    
    return PREDEFINED_SEEDS[:n_seeds]


def get_key_by_value(value,map):
    """
    Return the value for a given key in a dictionary
    """
    key = next((key for key,val in map.items() if val == value),None)
    return key

def invert_dict(dict):
    """
    Invert the keys and values of a dictionary
    """
    return {value: key for key, value in dict.items()}

def get_tasks(tasks):
    """
    Map the task argument to constant task sets or the single task if only one given
    """
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
        tasks = [tasks]
    return tasks
    
def get_max_len(max_length,task):
    """
    Map the max length argument to a max length used in the tokenizer
    """
    if max_length == "max":
        max_length = None
    elif max_length == "std":
        max_length = MAX_LENS[task]
    else:
        max_length = int(max_length)
    
    return max_length


def logger_setup():
    """
    Set up the logger
    
    """
    logger = logging.getLogger(__name__)
    
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
    return logger