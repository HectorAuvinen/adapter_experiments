def process_labels(labels):
    
    """
    Converts labels to integers and performs zero indexing if necessary.
    If all labels are already 0 or 1, no conversion is performed.
    
    Args:
    labels (List[Union[int, str]]): List of labels where each label can be an integer or a string.
    
    Returns:
    List[int]: Processed labels as integers with zero indexing.
    """
    # Check if all labels are 0 or 1
    all_01 = all(label in (0, 1) for label in labels)

    processed_labels = []
    for label in labels:
        if all_01:
            processed_labels.append(label)  # Leave labels as they are
        elif isinstance(label, int):
            processed_labels.append(label - 1)  # Convert to zero indexing
        elif isinstance(label, str) and label.isdigit():
            processed_labels.append(int(label) - 1)  # Convert to int and zero indexing
        else:
            processed_labels.append(0)  # Default to zero indexing if not recognized
    
    return processed_labels