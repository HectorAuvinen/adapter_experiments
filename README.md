# adapter_experiments

This repository allows for using the Adapters library to conduct experiments with bottleneck adapters.

### Results in [AdapterFusion: Non-Destructive Task Composition for Transfer Learning](https://arxiv.org/pdf/2005.00247.pdf) by Pfeiffer et al.(2021)

![AdapterFusion Results](./adapterfusion_results.JPG)

[paper](https://arxiv.org/pdf/2005.00247.pdf)
## Environment

There is an `environment.yml` file that contains the dependencies for the project. If you encounter problems in installing torch from `environment.yml`, create an empty environment and install `adapters`,`datasets`,`accelerate` and `evaluate`. This will be sufficient for the project.


## Usage

The core of this project is `runner.py`, which is used to conduct experiments with various tasks and models. The script accepts a number of arguments that allow customizing the experiments. All datasets can be downloaded from huggingface so you do not need them on your disk.

### Arguments

- `--task_name`: Specify the task or tasks to conduct the experiments on. See the list of supported tasks in the README.
- `--model_name`: Choose the model to use in the experiments. See the list of supported models in the README.
- `--output_path`: Set the output path for the experiment results, including model performances and configurations. Use the `--keep_checkpoints` flag to retain trained models.
- `--adapter_config_path`: Provide the path to the adapter configuration. This should be a JSON file with a dictionary of configuration names and their corresponding settings. See `configs` for example configurations.
- `--training_config_path`: Path to the Transformers training configuration JSON file. See `configs` for example configurations.
- `--logging`: Set the log level for the execution of the script.
- `--single_config`: Use this flag to run with only the first configuration entry if your JSON file contains multiple configurations.
- `--train_batch_size`: Define the training batch size. This can also be set in the training configuration file.
- `--eval_batch_size`: Set the evaluation batch size. Can be configured in the training configuration file as well.
- `--max_len`: Maximum sequence length to use. Use 'std' for predefined lengths, 'max' for no limit, or specify an integer value.
- `--eval_column`: Specify a custom column to use for evaluation, if you want to use something other than the default (`validation`) column.
- `--num_seeds`: Number of different seeds to use for running the experiments. Results are saved under the seed-named directories in the output path.
- `--early_stopping`: Set the tolerance for early stopping.
- `--keep_checkpoints`: Add this flag to save training checkpoints.
- `--mode`: Choose the training mode - either 'adapter', 'ft' (full fine-tuning), or 'all'.
- `--debug`: Activate debugging mode where datasets are sliced to contain only 10 samples.
- `--one_seed`: Run experiments with a single specified seed (int).

### Example Command

To run an experiment with `runner.py`, use the following command:

```bash
python runner.py --task_name "classification" --model_name "bert-base-uncased" --output_path "./outputs" --adapter_config_path "./src/configs/adapter_config.json" --training_config_path "./src/configs/training_config.json" --logging INFO --max_len "std" --num_seeds 3 --mode "all"
```


## Supported models

| Model        | Layers | Hidden Size | Num Heads | Num Params  |
|--------------|--------|-------------|-----------|-------------|
| RoBERTa-Tiny | 4      | 512         | 8         | 27,982,336  |
| BERT-Small   | 4      | 512         | 8         | 28,763,648  |
| BERT-Tiny    | 2      | 128         | 2         | 4,385,920   |


## Supported datasets
| Dataset | Category | Description | (Relevant) Columns | Samples Train | Samples Val| Samples Test |
|---------|----------|-------------|--------------------|---------------|-------|--------|
|MNLI| Natural language inference / sentence entailment| Given a sentence pair classify it as entailment, contradiction or neutral |'premise','hypothesis','label'|  392702|  9815|  9832|
|QQP| Sentence relatedness / sentence similarity| Given a question pair classify it as paraphrase or not |'question1','question2','label'|  363849|  40428|  390965|
|SST2| Sentiment analysis / binary classification| Given a sentence classify it as positive or negative |'sentence','label'|  67349|  872|  1821|
|Winogrande|  Commonsense reasoning / multiple choice| Given a sentence and blank fill in the blank |'sentence','option1','option2','answer'| 40398|1767|1267 |
|IMDB| Sentiment analysis / binary classification| Given a sentence classify it as positive or negative |'text','label'|  25000|  0|  25000|
|Hellaswag|Commonsense reasoning / multiple choice| Given a sentence choose the correct ending |'ctx_a','ctx_b','ctx','endings''label'| 39905|10003|10042 |
|SocialIQA| Commonsense reasoning / multiple choice| Given a sentence and question choose the correct answer |'context','question','answerA','answerB','answerC,'label'|  33410|  1954|  0|
|CosmosQA|  Commonsense reasoning / multiple choice| Given a sentence and question choose the correct answer |'context','question','answer0','answer1','answer2','answer3','label'|  25262|  6963|  2985|
|SciTail| Commonsense reasoning / multiple choice| TODO| TODO | 23097|1304 |2126|
|Argument mining| TODO| TODO |TODO|  18341|  2042|  5109|
|CSQA| TODO| TODO| TODO| 9741|1221|1140|
|BoolQ| Reading comprehesion| Given a question and a sentence answer yes or no |'question','answer','passage'|  9427|  3270|  0|
|MRPC| Sentence relatedness / sentence similarity| Given a sentence pair classify it as paraphrase or not |'sentence1','sentence2','label'|  3668|  408|  1725|
|SICK| Natural language inference / sentence entailment| Given a sentence pair classify it as entailment, contradiction or neutral |'sentence_A','sentence_B','label'|  4439|  495|  4906|
|RTE| Natural language inference / sentence entailment| Given a sentence pair classify the second sentence as entails or does not entail |'sentence1','sentence2','label'|  2490|  277|  3000|
|CB| Natural language inference / sentence entailment| Given a sentence pair classify it as entailment, contradiction or neutral |'premise','hypothesis','label'|  250|  277|  250|


