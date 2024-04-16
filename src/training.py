import numpy as np
from transformers import TrainingArguments,EvalPrediction,EarlyStoppingCallback,Trainer
from adapters import AdapterTrainer


class TrainingParameters:
    """
    Class for handling Transformers training arguments
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
        # save_total_limit=1,  # Save only the best model checkpoint
        # save_steps=1000,  # Save model checkpoints every 1000 steps (adjust as needed)
        self.lr_scheduler_type = lr_scheduler_type,  # Specify the learning rate scheduler type
        self.load_best_model_at_end=load_best_model_at_end,  # Load the best model checkpoint at the end
        self.metric_for_best_model = metric_for_best_model
        self.early_stopping_patience = early_stopping_patience
        self.save_total_limit = save_total_limit  
        # https://stackoverflow.com/questions/69087044/early-stopping-in-bert-trainer-instances
        # early stopping


def calculate_accuracy(p: EvalPrediction):
    """
    Calculate the accuracy for the evaluation
    """
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy":(preds == p.label_ids).mean()}

def get_training_arguments(args):
    """
    Instantiate the training arguments
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
        #callbacks= [EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience)]
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
        #callbacks= [EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience)]
    )
    return trainer