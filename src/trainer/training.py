import numpy as np
from transformers import TrainingArguments,EvalPrediction,EarlyStoppingCallback
from adapters import AdapterTrainer


class TrainingParameters:
    def __init__(self, label_names=["labels"],
                 evaluation_strategy="steps",
                 learning_rate=1e-4,
                 num_train_epochs=30,
                 per_device_train_batch_size=8,
                 per_device_eval_batch_size=8,
                 eval_steps=50,
                 logging_steps=200,
                 output_dir="/eval_results",
                 overwrite_output_dir=True,
                 remove_unused_columns=False,
                 lr_scheduler_type='linear',
                 load_best_model_at_end=True,
                 metric_for_best_model = "accuracy",
                 early_stopping_patience=3,
                 save_total_limit=5
                 ):
        self.label_names = label_names
        self.evaluation_strategy = evaluation_strategy
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
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy":(preds == p.label_ids).mean()}

def get_training_arguments(args):
    training_args = TrainingArguments(
        label_names=args.label_names,
        evaluation_strategy=args.evaluation_strategy,
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

def get_trainer(training_args,dataset,model,early_stopping=3):
    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=calculate_accuracy,
        callbacks= [EarlyStoppingCallback(early_stopping_patience=early_stopping)]
        #callbacks= [EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience)]
    )
    return trainer


"""
def get_training_arguments(args):
    training_args = TrainingArguments(
        label_names=args.get("label_names",""),
        evaluation_strategy=args.get("evaluation_strategy",""),
        learning_rate=args.get("learning_rate",""),
        num_train_epochs=args.get("num_train_epochs",""),
        per_device_train_batch_size = args.get("per_device_train_batch_size",""),
        per_device_eval_batch_size = args.get("per_device_eval_batch_size",""),
        eval_steps = args.get("eval_steps",""),
        logging_steps = args.get("logging_steps",""),
        output_dir = args.get("output_dir",""),
        overwrite_output_dir = args.get("overwrite_output_dir",""),
        remove_unused_columns = args.get("remove_unused_columns",""),
        lr_scheduler_type=args.get("lr_scheduler_type",""),
        load_best_model_at_end=args.get("load_best_model_at_end",""),
        metric_for_best_model=args.get("metric_for_best_model",""),
        save_total_limit=args.get("save_total_limit","")
    )
    return training_args

def get_trainer(training_args,dataset,model):
    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=calculate_accuracy,
        callbacks= [EarlyStoppingCallback(early_stopping_patience=training_args.get("early_stopping_patience"))]
    )
    return trainer    
"""