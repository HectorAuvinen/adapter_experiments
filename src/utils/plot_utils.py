import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the absolute path to the root of the project
project_root = os.path.dirname(script_dir)
# Add the project root to the sys.path
sys.path.insert(0, project_root)


from src.constants.constants import DATASET_SIZES


def prepare_eval_for_plots(new_results):
    # Calculate mean and standard deviation for accuracies
    mean_accuracies = {}
    std_dev_accuracies = {}  # Store standard deviations here

    for reduction_factor, seeds in new_results.items():
        for seed, tasks in seeds.items():
            for task, accuracy in tasks.items():
                if task not in mean_accuracies:
                    mean_accuracies[task] = {}
                    std_dev_accuracies[task] = {}  # Initialize std deviation dict
                if reduction_factor not in mean_accuracies[task]:
                    mean_accuracies[task][reduction_factor] = []
                    std_dev_accuracies[task][reduction_factor] = []  # Initialize list for std deviation values
                mean_accuracies[task][reduction_factor].append(accuracy)

    for task, reduction_factors in mean_accuracies.items():
        for reduction_factor, accuracies in reduction_factors.items():
            mean_accuracies[task][reduction_factor] = np.mean(accuracies)
            std_dev_accuracies[task][reduction_factor] = np.std(accuracies)  # Calculate standard deviation

    # Sort and prepare data for plotting
    # sorted_reduction_factors = sorted(new_results.keys(), key=lambda x: (x != 'full_fine_tune', int(x.split('_')[-1]) if x != 'full_fine_tune' else -1))
    def custom_sort_key(x):
        # Handle 'full fine tune' as a special case
        if x == 'full_fine_tune':
            return (0, 0)
        # Extract numerical part for sorting
        parts = x.split('_')
        # Assuming the format is 'output_adapter_redf_NUMBER'
        number_part = parts[-1]
        try:
            # Convert to integer for proper numerical comparison
            if number_part[0] == "0":
                number = float(f"{number_part[0]}.{number_part[1]}")
            else:
                number = int(number_part)
        except ValueError:
            # In case of any unexpected format, fallback to original string comparison
            number = number_part
        return (1, number)

    # Use the custom sorting function in your existing sorting line
    sorted_reduction_factors = sorted(new_results.keys(), key=custom_sort_key)
    """    DATASET_SIZES = {'mnli':392702, 'qqp':363849, 'sst2':67349, 'winogrande':40398, 'imdb':25000, 'hellaswag':39905,
                    'social_i_qa':33410, 'cosmos_qa':25262, 'scitail':23097, 'argument':18341,
                    'commonsense_qa':9741, 'boolq':9427, 'mrpc':3668, 'sick':4439, 'rte':2490, 'cb':250}
    """
    tasks = sorted([task for task in mean_accuracies.keys() if task in DATASET_SIZES], key=lambda task: DATASET_SIZES[task], reverse=True)

    data = {reduction_factor: [] for reduction_factor in sorted_reduction_factors}
    errors = {reduction_factor: [] for reduction_factor in sorted_reduction_factors}  # For standard deviations
    for task in tasks:
        for reduction_factor in sorted_reduction_factors:
            data[reduction_factor].append(mean_accuracies[task].get(reduction_factor, np.nan))
            errors[reduction_factor].append(std_dev_accuracies[task].get(reduction_factor, np.nan))
    
    return data,errors,sorted_reduction_factors,tasks

def plot_evaluation(data,errors,sorted_reduction_factors,tasks,plot_title):
    plt.figure(figsize=(12, 8))
    num_reduction_factors = len(sorted_reduction_factors)
    bar_width = 0.8 / num_reduction_factors
    #
    error_kw = {
    'capthick':1,
    'capsize': 2,  # Adjust cap size if needed
    'elinewidth': 0.5#bar_width / 2,  # Adjust line width to match bar width as closely as possible
    #'ecolor': 'lightgrey',  # Set error bar color to grey
    #'alpha': 0.7,  # Adjust transparency of the error bars
    }
    #
    for i, reduction_factor in enumerate(sorted_reduction_factors):
        positions = np.arange(len(tasks)) + i * bar_width
        plt.bar(positions, data[reduction_factor], width=bar_width, alpha=0.8, label=reduction_factor, 
                yerr=errors[reduction_factor], error_kw=error_kw,capsize=5)

    plt.yticks(np.arange(0, 1.05, 0.05))
    plt.xlabel('Task')
    plt.ylabel('Mean Accuracy')
    plt.title(f'Mean Accuracy by Task and Reduction Factor:{plot_title}')
    labels = [f"{task}\n{DATASET_SIZES[task]} training samples" for task in tasks]
    plt.xticks(np.arange(len(tasks)) + bar_width * (len(sorted_reduction_factors) - 1) / 2, labels,rotation=45)
    plt.grid(True,which="both",axis="y",linestyle="--",linewidth=0.5,color="grey",alpha=0.5)
    plt.legend(title="Reduction Factor", loc="lower right")
    plt.tight_layout()
    plt.show()