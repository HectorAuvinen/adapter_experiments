ALL_TASKS = [
    "cb","rte","sick","mrpc","boolq","commonsense_qa",
    "argument","scitail","cosmos_qa","social_i_qa",
    "hellaswag","imdb","winogrande","sst2","qqp","mnli"]

SUBSET_TASKS = [
    "cb","rte","sick","mrpc","boolq"]

CLF_TASKS = [
    "cb","rte","sick","mrpc","boolq",
    "argument","scitail","imdb","sst2","qqp","mnli"]   
    
    
MC_TASKS = [
    "commonsense_qa","cosmos_qa","social_i_qa",
    "hellaswag","winogrande"]

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

MODEL_MAP = {"bert-base-uncased":"bert-base-uncased",
             "bert-tiny-uncased":"google/bert_uncased_L-2_H-128_A-2",
             "bert-mini-uncased":"google/bert_uncased_L-4_H-256_A-4"}


PREDEFINED_SEEDS = [32, 18, 19, 42,512, 1111, 2048, 1234, 8192, 12345]