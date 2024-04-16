from constants import PREDEFINED_SEEDS

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