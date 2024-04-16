
# different group tasks
GLUE_TASKS = ['ax', 'cola', 'mnli', 'mnli_matched', 'mnli_mismatched', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli']
SUPER_GLUE_TASKS = ['axb', 'axg', 'boolq', 'cb', 'copa', 'multirc', 'record', 'rte', 'wic', 'wsc', 'wsc.fixed']

# tasks that can be loaded from disk
DISK_TASKS = {"argument": "../data/hf_data/argument_mining"}

# list of all supported tasks
ALL_TASKS = [
    "cb","rte","sick","mrpc","boolq","commonsense_qa",
    "argument","scitail","cosmos_qa","social_i_qa",
    "hellaswag","imdb","winogrande","sst2","qqp","mnli"]

# experiment round 1 tasks
SUBSET_TASKS = [
    "cb","rte","sick","mrpc","boolq"]

# experiment round 2 tasks
SUBSET_TASKS_2 = [
    "rte","boolq","argument","imdb","winogrande"
]

# experiment round 3 tasks (2 of each dataset size)
SUBSET_TASKS_3 = ["sick","rte","boolq","commonsense_qa","argument","cosmos_qa","winogrande","sst2"]

# experiment round 4 tasks (final tasks)
SUBSET_TASKS_4 = ["sick","sst2"]

# classification tasks
CLF_TASKS = [
    "cb","rte","sick","mrpc","boolq",
    "argument","scitail","imdb","sst2","qqp","mnli"]   
    
# multiple choice tasks    
MC_TASKS = [
    "commonsense_qa","cosmos_qa","social_i_qa",
    "hellaswag","winogrande"]

# max lengths used in the round 1 and 2 experiments (smaller max len for multiple choice due to memory issues)
MAX_LENS = {
    "cb":256,
    "rte":256,
    "sick":256,
    "mrpc":256,
    "boolq":256,
    "commonsense_qa":128,
    "argument":256,
    "scitail":256,
    "cosmos_qa":128,
    "social_i_qa":128,
    "hellaswag":128,
    "imdb":256,
    "winogrande":128,
    "sst2":256,
    "qqp":256,
    "mnli":256
}

# model map so that different models are easier to call in script arguments
MODEL_MAP = {"bert-base-uncased":"bert-base-uncased",
             "bert-tiny-uncased":"google/bert_uncased_L-2_H-128_A-2",
             "bert-mini-uncased":"google/bert_uncased_L-4_H-256_A-4",
             "bert-small-uncased":"google/bert_uncased_L-4_H-512_A-8",
             "distilroberta-tiny-cased": "sshleifer/tiny-distilroberta-base",
             "electra-small": "google/electra-small-discriminator",
             "albert-base": "albert/albert-base-v1",
             "t5-small": "google-t5/t5-small",
             "roberta-tiny":"haisongzhang/roberta-tiny-cased"}

# dataset name mapping for plotting
NAME_MAP = {'MNLI':"mnli", 'QQP':"qqp", 'SST':"sst2", 'WGrande':"winogrande", 
            'IMDB':"imdb", 'HSwag':"hellaswag", 'SocialIQA':"social_i_qa", 'CosQA':"cosmos_qa", 
            'SciTail':"scitail", 'Argument':"argument", 'CSQA':"commonsense_qa", 'BoolQ':"boolq",
            'MRPC':"mrpc", 'SICK':"sick", 'RTE':"rte", 'CB':"cb"}

# list of predefined seeds
PREDEFINED_SEEDS = [32, 18, 19, 42,512, 1111, 2048, 1234, 8192, 12345]

# all dataset sizes
DATASET_SIZES = {'mnli':392702, 'qqp':	363849, 'sst2':67349, 'winogrande':40398, 'imdb':25000, 'hellaswag':39905,
    'social_i_qa':33410, 'cosmos_qa':25262, 'scitail':23097, 'argument':18341,
    'commonsense_qa':9741, 'boolq':9427, 'mrpc':3668, 'sick':4439, 'rte':2490, 'cb':250}

# Adapterfusion paper results for ST-A (reduction factor 2)
PAPER_RESULTS_REDF_2 = {
    'Dataset': [
        'MNLI', 'QQP', 'SST', 'WGrande', 'IMDB', 'HSwag', 'SocialIQA', 'CosQA', 'SciTail', 
        'Argument', 'CSQA', 'BoolQ', 'MRPC', 'SICK', 'RTE', 'CB'
    ],
    'ST-A': [
        84.60, 90.57, 92.66, 62.11, 94.20, 39.45, 60.95, 59.32, 94.44,
        76.83, 57.83, 77.14, 86.13, 87.50, 70.68, 87.85
    ]
}

# Adapterfusion paper results for ST-A (reduction factor 16)
PAPER_RESULTS_REDF_16 = {
    'Dataset': [
        'MNLI', 'QQP', 'SST', 'WGrande', 'IMDB', 'HSwag', 'SocialIQA', 'CosQA', 'SciTail', 
        'Argument', 'CSQA', 'BoolQ', 'MRPC', 'SICK', 'RTE', 'CB'
    ],
    'ST-A': [
        84.32, 90.59, 91.85, 61.09, 93.85, 38.11, 62.41, 60.01, 93.90,
        77.65, 58.91, 75.66, 85.16, 86.20, 71.04, 86.07
    ]
}