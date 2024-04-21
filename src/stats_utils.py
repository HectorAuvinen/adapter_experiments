from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt

from .constants import DATASET_SIZES,HIDDEN_SIZES
from .file_utils import read_eval_results_2


def sort_results(results,task,model_name,hidden_sizes):
    adapter_sizes = []
    performances = []

    for redf, data in results[task][model_name].items():
        if redf != 'fft':
            adapter_sizes.append(hidden_sizes[model_name]/float(redf))
            performances.append(data['mean_accuracy'])
    return adapter_sizes,performances



def calculate_dataset_correlation(path,dataset_sizes):
    root_folder = Path(path)
    results = read_eval_results_2(root_folder)
    
    mean_accuracies = {}

    for dataset in results:
        mean_accuracies[dataset] = {}
        for model in results[dataset]:
            accuracies = []
            for adapter_size in results[dataset][model]:
                if adapter_size != 'fft':
                    accuracies.append(results[dataset][model][adapter_size]['mean_accuracy'])
            # Calculate the overall mean accuracy for the current model, excluding fft
            if accuracies: 
                mean_accuracies[dataset][model] = np.mean(accuracies)

    df = pd.DataFrame(columns=["Model", "Spearman's rank correlation coefficient", "p-value"])
    
    # mean_accuracies now contains the mean accuracy for each model (no fft), for each dataset
    for model in ['bert-tiny-uncased', 'roberta-tiny']:
        dataset_sizes = []
        performances = []

        # Collect dataset sizes and performances for the current model
        for dataset, models_performance in mean_accuracies.items():
            if dataset in DATASET_SIZES and model in models_performance:
                dataset_sizes.append(DATASET_SIZES[dataset])
                performances.append(models_performance[model])

        corr, p_value = spearmanr(dataset_sizes, performances)
        
        new_row = pd.Series({'Model': model, "Spearman's rank correlation coefficient": corr, 'p-value': p_value})
        df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
    
    return df


def calculate_correlation(model_name,task,hidden_sizes,results):
    """
    Calculate Spearman's correlation coefficient for adapter sizes and corresponding performance results
    """
    adapter_sizes,performances = sort_results(results,task,model_name,hidden_sizes)

    paired_sorted = sorted(zip(adapter_sizes,performances),key=lambda x:x[0])
    sorted_adapter_sizes,sorted_performances = zip(*paired_sorted)

    corr, p_value = spearmanr(sorted_adapter_sizes, sorted_performances)
    # print(f"Spearman's Rank Correlation: Correlation={corr}, p-value={p_value}")
    
    return corr,p_value


def anova_test(model_name,task,hidden_sizes,results,bins=[0,100,1000,10000],show_plot=True):
    """
    Conduct ANOVA and Tukey's HSD test on the different adapter size bins
    """
    # get lists of sizes and performances and create a df from this
    adapter_sizes,performances = sort_results(results,task,model_name,hidden_sizes)
    
    data = pd.DataFrame({'Adapter Size': adapter_sizes, 'Performance': performances})

    # Define the bins for the adapter size categories
    labels = ['Small', 'Medium', 'Large']
    data['Size Category'] = pd.cut(data['Adapter Size'], bins=bins, labels=labels)

    # ANOVA
    anova = stats.f_oneway(data[data['Size Category'] == 'Small']['Performance'],
                        data[data['Size Category'] == 'Medium']['Performance'],
                        data[data['Size Category'] == 'Large']['Performance'])
    
    # if p-value small enough, perform Tukey's HSD
    if anova.pvalue < 0.05:
        print("Significant differences found, proceeding with Tukey's HSD test.")
        mc = pairwise_tukeyhsd(endog=data['Performance'], groups=data['Size Category'], alpha=0.05)
        
        if show_plot:
            mc.plot_simultaneous()
            plt.show()
        
    else:
        print("No significant differences found. No need for post-hoc analysis.")
       
    return anova,bins


def calculate_stats_from_path(path,skip=None):
    root_folder = Path(path)
    results = read_eval_results_2(root_folder)
    df = pd.DataFrame({"model":[],"task":[],"corr":[],"p-value":[]})
    for task in ["sick","sst2"]:
        for model_name in HIDDEN_SIZES.keys():
            if skip and model_name == skip:
                continue
            corr,p_val = calculate_correlation(model_name=model_name,task=task,hidden_sizes=HIDDEN_SIZES,results=results)
            new_row = pd.Series({'model': model_name, 'task': task, 'corr': corr, 'p-value': p_val})
            df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
    return df

def calculate_anova_from_path(path,skip=None,bins=[0,100,2000,10000]):
    root_folder = Path(path)
    results = read_eval_results_2(root_folder)
    df = pd.DataFrame({"model":[],"task":[],"f-statistic":[],"p-value":[],"bins":[]})
    for task in ["sick","sst2"]:
        for model_name in HIDDEN_SIZES.keys():
            if skip and model_name == skip:
                continue
            anova,bins = anova_test(model_name=model_name,task=task,hidden_sizes=HIDDEN_SIZES,results=results,bins=bins)
            new_row = pd.Series({'model': model_name, 'task': task, 'f-statistic': anova.statistic, 'p-value': anova.pvalue,"bins":bins})
            df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
    return df