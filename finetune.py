from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate

class Finetuner:
    def __init_(self,
                dataset_name="",
                model_name="Intel/dynamic_tinybert"):
        self.dataset_name = dataset_name # e.g. yelp_review_full
        self.model_name = model_name # e.g. google-bert/bert-base-cased

        # load dataset
        dataset = load_dataset(dataset_name)
        
        # tokenize
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenized_datasets = dataset.map(self.tokenize_function, batched=True)

        # smaller subset for testing/debugging
        self.small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
        self.small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

        # load mdoel
        # todo: replace with QA model type
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        

    def tokenize_function(self,examples):
        return self.tokenizer(examples["text"], padding="max_length", truncation=True)


    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
        

    def train(self):
        training_args = TrainingArguments(output_dir="test_trainer")

        # evaluation metric
        metric = evaluate.load("accuracy")

        # training 
        training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.small_train_dataset,
            eval_dataset=self.small_eval_dataset,
            compute_metrics=compute_metrics,
        )

        self.trainer.train()

def main():
    dataset_path = ""
    finetuner = Finetuner(dataset_path)
    finetuner.train()

