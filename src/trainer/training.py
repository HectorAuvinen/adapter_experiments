import numpy as np
from transformers import TrainingArguments,EvalPrediction
from adapters import AdapterTrainer

def calculate_accuracy(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    return {"acc":(preds == p.label_ids).mean()}

def get_training_arguments(args):
    training_args = TrainingArguments(
        label_names=["labels"],
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size = args.per_device_train_batch_size,
        per_device_eval_batch_size = args.per_device_eval_batch_size,
        logging_steps = args.logging_steps,
        overwrite_output_dir = args.overwrite_output_dir,
        remove_unused_columns = args.remove_unused_columns
    )
    return training_args
    
def get_trainer(training_args,dataset,model):
    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=calculate_accuracy
        
    )