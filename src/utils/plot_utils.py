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
from src.utils.file_utils import *


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
        print("number",number)
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

def plot_evaluation_subplots(data, errors, sorted_reduction_factors, tasks, plot_title,suffix=""):
    num_reduction_factors = len(sorted_reduction_factors)
    bar_width = 0.8 / num_reduction_factors
    error_kw = {'capthick':1, 'capsize': 2, 'elinewidth': 0.5}

    for i, reduction_factor in enumerate(sorted_reduction_factors):
        positions = np.arange(len(tasks)) + i * bar_width
        plt.bar(positions, data[reduction_factor], width=bar_width, alpha=0.8, label=reduction_factor, 
                yerr=errors[reduction_factor], error_kw=error_kw, capsize=5)

    plt.yticks(np.arange(0, 1.05, 0.05))
    plt.xlabel('Task')
    plt.ylabel('Mean Accuracy')
    plt.title(f'{plot_title} {suffix}')
    labels = [f"{task}\n{DATASET_SIZES[task]} training samples" for task in tasks]
    plt.xticks(np.arange(len(tasks)) + bar_width * (len(sorted_reduction_factors) - 1) / 2, labels, rotation=45)
    plt.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5, color="grey", alpha=0.5)
    plt.legend(title="Reduction Factor", loc="lower left")

def plot_all_models(root_folder,suffix=""):
    
    root_path = Path(root_folder)
    model_folders = [f for f in root_path.iterdir() if f.is_dir()]

    n = len(model_folders)
    grid_size = int(np.ceil(np.sqrt(n)))

    plt.figure(figsize=(12 * grid_size, 8 * grid_size))
    
    for idx, model_folder in enumerate(model_folders, start=1):
        print(f"Processing: {model_folder.name}")
        new_results = read_eval_results(model_folder, two_datasets=True)
        #print(f"Batch size: {batch_size}")
        #print(f"Max length: {max_len}")
        data, errors, sorted_reduction_factors, tasks = prepare_eval_for_plots(new_results)
        
        plt.subplot(grid_size, grid_size, idx)
        plot_evaluation_subplots(data, errors, sorted_reduction_factors, tasks, model_folder.name,suffix=suffix)

    plt.tight_layout()
    plt.show() 

def plot_line_results_grid(results,hidden_sizes,marker):
    num_datasets = len(results)

    fig, axes = plt.subplots(1, num_datasets, figsize=(24, 7))
    
    if num_datasets == 1:
        axes = [axes]
    
    for ax, (dataset_name, dataset_results) in zip(axes, results.items()):
        ax.set_xscale('log')

        for model_name, model_results in dataset_results.items():
            adapter_sizes = []
            mean_accuracies = []
            std_devs = []
            full_fine_tune_acc = None
            full_fine_tune_std = None

            if 'fft' in model_results:
                full_fine_tune_acc = model_results['fft']['mean_accuracy']
                full_fine_tune_std = model_results['fft']['std_dev']

            non_fft_data = [(float(rf), stats['mean_accuracy'], stats['std_dev'])
                            for rf, stats in model_results.items() if rf != 'fft']
            non_fft_data.sort(key=lambda x: x[0])

            for rf, mean_acc, std_dev in non_fft_data:
                adapter_size = hidden_sizes[model_name] / rf
                adapter_sizes.append(adapter_size)
                mean_accuracies.append(mean_acc)
                std_devs.append(std_dev)
                
                if model_name == marker:
                    ax.text(adapter_size, mean_acc, f'{adapter_size:.0f}', fontsize=8, va="bottom", ha='center')

            ax.errorbar(adapter_sizes, mean_accuracies, yerr=std_devs, fmt='-o', label=model_name,
                        alpha=0.7, elinewidth=2, capsize=5)

            if full_fine_tune_acc is not None:
                min_x, max_x = ax.get_xlim()
                min_x = 0  # or set to the minimal adapter size explicitly if needed
                ax.hlines(full_fine_tune_acc, min_x, max_x, colors=ax.lines[-1].get_color(),
                          linestyles='--', alpha=0.5)
                ax.errorbar(min_x, full_fine_tune_acc, yerr=full_fine_tune_std, fmt='o',
                            color=ax.lines[-1].get_color(), alpha=0.7, elinewidth=2, capsize=5)

        ax.set_title(f'Performance on {dataset_name}')
        ax.set_xlabel('Adapter Size')
        ax.set_ylabel('Mean Accuracy')
        ax.legend(loc="lower right")
        ax.grid(True, which="both", linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()


def plot_line_results(results, hidden_sizes,marker):
    for dataset_name, dataset_results in results.items():
        plt.figure(figsize=(12, 7))

        # Set x-axis to logarithmic scale
        plt.xscale('log')

        for model_name, model_results in dataset_results.items():
            adapter_sizes = []
            mean_accuracies = []
            std_devs = []
            full_fine_tune_acc = None
            full_fine_tune_std = None

            # Extract full fine-tune accuracy and deviation first
            if 'fft' in model_results:
                full_fine_tune_acc = model_results['fft']['mean_accuracy']
                full_fine_tune_std = model_results['fft']['std_dev']

            # Extract other reduction factors and sort by adapter size
            non_fft_data = [(float(rf), stats['mean_accuracy'], stats['std_dev'])
                            for rf, stats in model_results.items() if rf != 'fft']
            non_fft_data.sort(key=lambda x: x[0])

            for rf, mean_acc, std_dev in non_fft_data:
                adapter_size = hidden_sizes[model_name] / rf
                adapter_sizes.append(adapter_size)
                mean_accuracies.append(mean_acc)
                std_devs.append(std_dev)
                
                if model_name == marker:
                    plt.text(adapter_size, mean_acc, f'{adapter_size:.0f}', fontsize=8,va="bottom", ha='center')

            # Plot line for model's reduction factors
            plt.errorbar(adapter_sizes, mean_accuracies, yerr=std_devs, fmt='-o', label=model_name,
                         alpha=0.7, elinewidth=2, capsize=5)

            # Plot the full fine-tune baseline as a horizontal line with vertical error bar
            if full_fine_tune_acc is not None:
                min_x, max_x = plt.xlim()
                min_x = 0
                plt.hlines(full_fine_tune_acc, min_x, max_x, colors=plt.gca().lines[-1].get_color(),
                           linestyles='--', alpha=0.5)
                plt.errorbar(min_x, full_fine_tune_acc, yerr=full_fine_tune_std, fmt='o',
                             color=plt.gca().lines[-1].get_color(), alpha=0.7, elinewidth=2, capsize=5)

        plt.title(f'Performance on {dataset_name}')
        plt.xlabel('Adapter Size')
        plt.ylabel('Mean Accuracy')
        plt.legend(loc="lower right")
        plt.grid(True, which="both", linestyle='--', alpha=0.5)
        plt.show()

    
def plot_baselines(results):
    """
    plot baseline results from the paper. Requires the following format:
    data = {
        'Dataset': [
            'MNLI', 'QQP', 'SST', 'WGrande', 'IMDB', 'HSwag', 'SocialIQA', 'CosQA', 'SciTail', 
            'Argument', 'CSQA', 'BoolQ', 'MRPC', 'SICK', 'RTE', 'CB'
        ],
        'ST-A': [
            84.60, 90.57, 92.66, 62.11, 94.20, 39.45, 60.95, 59.32, 94.44,
            76.83, 57.83, 77.14, 86.13, 87.50, 70.68, 87.85
        ]
    }
    """
    # Plotting
    plt.figure(figsize=(10,8))
    bars = plt.bar(results['Dataset'], results['ST-A'], color='skyblue')
    plt.xlabel('Dataset')
    plt.ylabel('ST-A Score')
    plt.title('ST-A Results for Datasets')

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, round(yval, 2), va='bottom', ha='center')

    plt.xticks(rotation=45)  
    plt.tight_layout()  

    # Show the plot
    plt.show()
    
    
def plot_baseline_reproduction(df,new_results):
    df['New-ST-A'] = df['Dataset'].map(new_results)  

    print(df)

    plt.figure(figsize=(12, 8))

    bar_width = 0.35

    r1 = np.arange(len(df['Dataset']))

    r2 = [x + bar_width for x in r1]

    bars1 = plt.bar(r1, df['ST-A'], color='skyblue', width=bar_width, label='ST-A Original')
    bars2 = plt.bar(r2, df['New-ST-A'], color='orange', width=bar_width, label='ST-A Ours')

    plt.xlabel('Dataset')
    plt.ylabel('Accuracy')
    plt.title('ST-A Reproduction Results')

    for bars in (bars1, bars2):
        for bar in bars:
            yval = bar.get_height()
            if not np.isnan(yval): 
                plt.text(bar.get_x() + bar.get_width() / 2.0, yval, round(yval, 2), va='bottom', ha='center',rotation=45)

    plt.xticks([r + bar_width/2 for r in range(len(df['Dataset']))], df['Dataset'], rotation=45)

    plt.legend(loc="lower right")
    plt.tight_layout()

    plt.show()