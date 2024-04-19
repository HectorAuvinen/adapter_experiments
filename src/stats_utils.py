from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import shapiro,pearsonr, spearmanr
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols

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
    #results = read_eval_results(root_folder)
    
    mean_accuracies = {}

    for dataset in results:
        mean_accuracies[dataset] = {}
        for model in results[dataset]:
            accuracies = []
            for adapter_size in results[dataset][model]:
                if adapter_size != 'fft':
                    accuracies.append(results[dataset][model][adapter_size]['mean_accuracy'])
            # Calculate the overall mean accuracy for the current model, excluding 'fft'
            if accuracies:  # Check if there are any adapter sizes besides 'fft'
                mean_accuracies[dataset][model] = np.mean(accuracies)

    df = pd.DataFrame(columns=["Model", "Spearman's rank correlation coefficient", "p-value"])
    
    # mean_accuracies now contains the mean accuracy for each model, excluding fft, for each dataset
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
    """    # Extract adapter sizes and performances
    adapter_sizes = []
    performances = []

    for redf, data in results[task][model_name].items():
        if redf != 'fft':  # Exclude the full fine-tune ('fft') case
            adapter_sizes.append(hidden_sizes[model_name]/float(redf))
            performances.append(data['mean_accuracy'])"""
    adapter_sizes,performances = sort_results(results,task,model_name,hidden_sizes)
    #print("Adapter Sizes:", adapter_sizes)
    #print("Performances:", performances)

    paired_sorted = sorted(zip(adapter_sizes,performances),key=lambda x:x[0])
    sorted_adapter_sizes,sorted_performances = zip(*paired_sorted)

    # Check normality for sorted adapter sizes and performances
    stat_sizes, p_sizes = shapiro(sorted_adapter_sizes)
    #print(f'Adapter Sizes Normality: Statistics={stat_sizes:.3f}, p-value={p_sizes:.3f}')
    #print(f'Adapter Sizes Normality: Statistics={stat_sizes:.5f}, p-value={p_sizes:.5f}')
    
    # Normality check for performances
    stat_perf, p_perf = shapiro(sorted_performances)
    #print(f'Performances Normality: Statistics={stat_perf:.5f}, p-value={p_perf:.5f}')

    #if p_sizes > 0.05 and p_perf > 0.05:
        # Both datasets appear normally distributed
        #corr, p_value = pearsonr(sorted_adapter_sizes, sorted_performances)
        #print(f"Pearson's Correlation: Correlation={corr:.5f}, p-value={p_value:.5f}")
    #else:
    # At least one of the datasets is not normally distributed
    corr, p_value = spearmanr(sorted_adapter_sizes, sorted_performances)
    # print(f"Spearman's Rank Correlation: Correlation={corr}, p-value={p_value}")
    
    return corr,p_value


def anova_test(model_name,task,hidden_sizes,results,bins=[0,100,1000,10000]):
    adapter_sizes,performances = sort_results(results,task,model_name,hidden_sizes)
    #adapter_sizes_poly = np.column_stack((np.ones(len(adapter_sizes)), adapter_sizes, np.power(adapter_sizes, 2)))
    #model = sm.OLS(performances, adapter_sizes_poly).fit()
    #print(model.summary())
    
    data = pd.DataFrame({'Adapter Size': adapter_sizes, 'Performance': performances})

    # Define the bins for the adapter size categories
    #bins = [0, 50,1000,2000]
    labels = ['Small', 'Medium', 'Large']
    data['Size Category'] = pd.cut(data['Adapter Size'], bins=bins, labels=labels)

    # tukey_results = pairwise_tukeyhsd(endog=data['Performance'], groups=data['Size Category'], alpha=0.05)
    # print("tykey results:",tukey_results)
    # Perform ANOVA
    anova = stats.f_oneway(data[data['Size Category'] == 'Small']['Performance'],
                        data[data['Size Category'] == 'Medium']['Performance'],
                        data[data['Size Category'] == 'Large']['Performance'])
    # new
    if anova.pvalue < 0.05:
        print("Significant differences found, proceeding with Tukey's HSD test.")
        
        # Prepare data for Tukey's HSD
        mc = pairwise_tukeyhsd(endog=data['Performance'], groups=data['Size Category'], alpha=0.05)
        #print(model_name,task)
        #print(mc)
        
        # For visualizing Tukey's HSD results
        # mc.plot_simultaneous()
        # plt.show()
        
    else:
        print("No significant differences found. No need for post-hoc analysis.")
        mc = None
       
    return anova,bins#,mc


def calculate_stats_from_path(path,skip=None):
    #root_folder = Path("F:/jku/practical_work/after_5th_march/reruns/trainable_params")  # Adjust this path
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
            #print("*"*100)
            #mc.plot_simultaneous() if mc else print("nothing to plot")
            #plt.show()
    return df