# Adapter_experiments

This repository allows for using the Adapters library to conduct experiments with bottleneck adapters.

## Environment

There is an `requirements.txt` file that contains the dependencies for the project.

```bash
python -m venv myenv
# On Windows
myenv\Scripts\activate
# On MacOS/Linux
source myenv/bin/activate

pip install -r requirements.txt
```


If you encounter problems in installing the requirements, create an empty environment and install `adapters`,`datasets`,`accelerate` and `evaluate`:

```bash
python -m venv myenv

# On Windows
myenv\Scripts\activate
# On MacOS/Linux
source myenv/bin/activate

pip install adapters datasets accelerate evaluate
```

This will be sufficient for the experiment runner. For the analyses in ```notebooks/result_analysis.ipynb```, you will need plotting and statistics packages. See ```requirements.txt``` for the list of requirements for manual installation.

## Project Structure
Here is an overview of the project structure:
```bash
ADAPTER_EXPERIMENTS/
│
├── configs/ # Configuration files for adapters and transformers models
├── data/ # Five datasets for local experiments
│
├── notebooks/ # notebooks for data preprocessing and analysis
│ ├── argument_data_preprocessing.ipynb # preprocessing the argument dataset
│ ├── data_preprocessing.ipynb # loading and saving the datasets on disk
│ ├── params_check.ipynb # checking the architectures of the different models
│ └── result_analysis.ipynb # plotting experiment results, doing statistical tests
│
├── outputs/ # output files from model runs
│
├── src/ 
│ ├──init.py
│ ├── constants.py # constants used across the codebase
│ ├── experiments.py # configuration and runner classes for conducting experiments
│ ├── file_utils.py # utilities for file operations
│ ├── load_data.py # module for handling and preparing the datasets
│ ├── model_setup.py # module for setting up the models and adapters
│ ├── plot_utils.py # plotting utilities
│ ├── preprocessing.py # module for tokenizers and transformations
│ ├── result_repository.py # class for handling model weight and result CRUD operations
│ ├── runner.py # main script for collecting arguments and triggering the experiments
│ ├── stats_utils.py # module for statistical tests
│ ├── training.py # module for training parameters and Hugging Face trainer
│ ├── utils.py # miscellaneous utilities
```

## Usage

The interface to this project is `runner.py`, which is used to conduct experiments with various tasks and models. The script accepts a number of arguments that allow customizing the experiments. All datasets can be downloaded from huggingface so you do not need them on your disk.

### Arguments

- `--task_name`: Specify the task or tasks to conduct the experiments on. See the list of supported tasks below.
- `--model_name`: Choose the model to use in the experiments. See the list of supported models below.
- `--output_path`: Set the output path for the experiment results, including model performances and configurations. Use the `--keep_checkpoints` flag to retain trained models.
- `--adapter_config_path`: Provide the path to the adapter configuration. This should be a JSON file with a dictionary of configuration names and their corresponding settings. See `configs` for example configurations.
- `--training_config_path`: Path to the Transformers training configuration JSON file. See `configs/training_config.json` for an example configuration.
- `--single_config`: Use this flag to run with only the first configuration entry if your JSON file contains multiple configurations.
- `--train_batch_size`: Define the training batch size. This can also be set in the training configuration file.
- `--eval_batch_size`: Set the evaluation batch size. This can also be set in the training configuration file.
- `--max_len`: Maximum sequence length to use. Use 'std' for predefined lengths (see ```src/constants.py```), 'max' for no limit, or specify an integer value.
- `--eval_column`: Specify a custom column to use for evaluation, if you want to use something other than the default column (which is `validation`).
- `--num_seeds`: Number of different seeds to use for running the experiments. Results are saved under seed-titled directories in the output path.
- `--early_stopping`: Set the tolerance for early stopping.
- `--keep_checkpoints`: Add this flag to save training checkpoints.
- `--mode`: Choose the training mode - either 'adapter', 'ft' (full fine-tuning), or 'all'.
- `--debug`: Activate debugging mode where datasets are sliced to contain only 10 samples.
- `--one_seed`: Run experiments with a single specified seed (int).

### Example Command

To run an experiment with `runner.py`, use the following command:

```bash
python runner.py --task_name "subset_4" --model_name "bert-tiny-uncased" --output_path "../outputs" --adapter_config_path "../configs/pfeiffer_128.json" --training_config_path "../configs/training_config.json" --max_len 256 --mode "all"
```

This will reproduce the results of the trainable parameters (Pfeiffer) experiments for bert-tiny seen in ```notebooks/result_analysis.ipynb```.

## Supported models

The supported models are specified in the table below. The ```model_name``` argument of ```runner.py``` expects the model name as it is in the ```ID``` column.

| Model        | ID | Layers | Hidden Size | Num Heads | Num Params  |
|--------------|----|--------|-------------|-----------|-------------|
| RoBERTa-Tiny |roberta-tiny| 4      | 512         | 8         | 27,982,336  |
| BERT-Small   |bert-small-uncased| 4      | 512         | 8         | 28,763,648  |
| BERT-Tiny    |bert-tiny-uncased| 2      | 128         | 2         | 4,385,920   |


## Supported datasets

All 16 datasets from the [Adapterfusion paper](https://arxiv.org/pdf/2005.00247.pdf) are supported. The ```task_name``` argument of ```runner.py``` expects the dataset name as it is in the ```ID``` column. The datasets are downloaded from Hugging Face (except for Argument). If you need to be able to run everything locally, use the local preprocessed datasets in ```data/hf_data``` (or use your own datasets). The local datasets need to be specified in ```DISK_TASKS``` in ```src/constants.py```.

| Dataset | ID | Samples Train | Samples Val| Samples Test |
|---------|----------|-------------|--------------------|---------------|
|MNLI| mnli |  392702|  9815|  9832|
|QQP| qqp |  363849|  40428|  390965|
|SST2| sst2 |  67349|  872|  1821|
|Winogrande|  winogrande | 40398|1767|1267 |
|IMDB| imdb|  25000|  0|  25000|
|Hellaswag|hellaswag| 39905|10003|10042 |
|SocialIQA| social_i_qa|  33410|  1954|  0|
|CosmosQA|  cosmos_qa|  25262|  6963|  2985|
|SciTail| scitail| 23097|1304 |2126|
|Argument| argument|  18341|  2042|  5109|
|CSQA| commonsense_qa| 9741|1221|1140|
|BoolQ| boolq|  9427|  3270|  0|
|MRPC| mrpc|  3668|  408|  1725|
|SICK| sick|  4439|  495|  4906|
|RTE| rte |  2490|  277|  3000|
|CB| cb |  250|  277|  250|
