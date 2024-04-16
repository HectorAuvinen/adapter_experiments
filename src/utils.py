from constants import PREDEFINED_SEEDS,ALL_TASKS,SUBSET_TASKS,SUBSET_TASKS_2,SUBSET_TASKS_3,SUBSET_TASKS_4,CLF_TASKS,MC_TASKS,MAX_LENS

def get_seeds(n_seeds):
    """
    Returns a consistent list of seeds based on number of seeds.
    
    """

    if n_seeds > len(PREDEFINED_SEEDS):
        raise ValueError(f"Requested number of seeds exceeds the limit of {len(PREDEFINED_SEEDS)}")
    
    # Slice the PREDEFINED_SEEDS list to get the desired number of seeds
    return PREDEFINED_SEEDS[:n_seeds]


def get_key_by_value(value,map):
    for key, val in map.items():
        if val == value:
            return key
    return None

def invert_dict(dict):
    return {value: key for key, value in dict.items()}

def get_tasks(tasks):
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
    
def get_max_len(max_length,task):
    if max_length == "max":
        max_length = None
    elif max_length == "std":
        max_length = MAX_LENS[task]
    else:
        max_length = int(max_length)