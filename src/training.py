import numpy as np
from transformers import TrainingArguments,EvalPrediction,EarlyStoppingCallback,Trainer
from adapters import AdapterTrainer


class TrainingParameters:
    """
    Class for handling Transformers training arguments
    
    Args:
        label_names (list): list of columns to be used as labels
        evaluation strategy (str): whether to use epochs or steps in evaluation
        save_strategy (str): whether to save based on epochs or steps
        learning_rate (float): learning rate for training
        num_train_epochs (int): number of training epochs
        per_device_train_batch_size (int): training batch size
        per_device_eval_batch_size (int): evaluation batch size
        eval_steps (int): number of evaluation epochs to run
        logging_steps (int): interval for logging performance
        output_dir (str): path to directory for outputs
        overwrite_output_dir (bool): if contents in output directory can be overwritten
        remove_unused_columns (bool): if unused columns can be remove from the data
        lr_scheduler_type (transformers.trainer_utils.SchedulerType): type of learning rate scheduler (e.g. linear)
        load_best_model_at_end (bool): whether to load the best model at end of training
        early_stopping_patience (int): how many epochs to wait for until stopping training if performance does not improve
        save_total_limit (int): how many models can be saved on disk simultaneously
    """
    def __init__(self, label_names=["labels"],
                 evaluation_strategy="epoch",
                 save_strategy="epoch",
                 learning_rate=1e-4,
                 num_train_epochs=30,
                 per_device_train_batch_size=8,
                 per_device_eval_batch_size=8,
                 eval_steps=1,
                 logging_steps=1000,
                 output_dir="/eval_results",
                 overwrite_output_dir=True,
                 remove_unused_columns=False,
                 lr_scheduler_type='linear',
                 load_best_model_at_end=True,
                 metric_for_best_model = "eval_accuracy",
                 early_stopping_patience=3,
                 save_total_limit=2
                 ):
        self.label_names = label_names
        self.evaluation_strategy = evaluation_strategy
        self.save_strategy = save_strategy
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps
        self.output_dir = output_dir
        self.overwrite_output_dir = overwrite_output_dir
        self.remove_unused_columns = remove_unused_columns
        self.lr_scheduler_type = lr_scheduler_type
        self.load_best_model_at_end=load_best_model_at_end
        self.metric_for_best_model = metric_for_best_model
        self.save_total_limit = save_total_limit
        self.early_stopping_patience = early_stopping_patience
        
    def change_batch_size(self,batch_size:int):
            self.per_device_train_batch_size = train_batch_size

    @staticmethod
    def get_training_arguments(args:"TrainingParameters") -> TrainingArguments:
        """
        Instantiate the training arguments for the Transformers Trainer class.
        
        Arguments:
            args (TrainingParameters): instance of TrainingParameters class
        
        Returns:
            training_args (transformers.TrainingArguments): instance of TrainingArguments class
        """
        training_args = TrainingArguments(
            label_names=args.label_names,
            evaluation_strategy=args.evaluation_strategy,
            save_strategy=args.save_strategy,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size = args.per_device_train_batch_size,
            per_device_eval_batch_size = args.per_device_eval_batch_size,
            eval_steps = args.eval_steps,
            logging_steps = args.logging_steps,
            output_dir = args.output_dir,
            overwrite_output_dir = args.overwrite_output_dir,
            remove_unused_columns = args.remove_unused_columns,
            lr_scheduler_type=args.lr_scheduler_type,
            load_best_model_at_end=args.load_best_model_at_end,
            metric_for_best_model=args.metric_for_best_model,
            save_total_limit=args.save_total_limit
        )
        return training_args

def calculate_accuracy(p: EvalPrediction):
    """
    Calculate the accuracy for the evaluation
    """
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy":(preds == p.label_ids).mean()}
    

def get_trainer(training_args,dataset,model,early_stopping=3,custom_eval=None):
    """
    Get the transformers trainer for an adapter
    """
    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if not custom_eval else dataset[custom_eval],
        compute_metrics=calculate_accuracy,
        callbacks= [EarlyStoppingCallback(early_stopping_patience=early_stopping)] if early_stopping else None
    )
    return trainer

def get_ft_trainer(training_args,dataset,model,early_stopping=3,custom_eval=None):
    """
    Get the transformers trainer for full fine-tuning
    """
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if not custom_eval else dataset[custom_eval],
        compute_metrics=calculate_accuracy,
        callbacks= [EarlyStoppingCallback(early_stopping_patience=early_stopping)] if early_stopping else None
    )
    return trainer