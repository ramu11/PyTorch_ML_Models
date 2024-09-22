
"""
In this section we will use as an example the MRPC (Microsoft Research Paraphrase Corpus) dataset, introduced in a paper by William B. Dolan and Chris Brockett. 
The dataset consists of 5,801 pairs of sentences, with a label indicating if they are paraphrases or not

refer HF doc: https://huggingface.co/learn/nlp-course/chapter3/2?fw=pt

The 🤗 Datasets library provides a very simple command to download and cache a dataset on the Hub. 
We can download the MRPC dataset like this:

pip install datasets

"""

from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
raw_datasets

# We can access each pair of sentences in our raw_datasets object by indexing, like with a dictionary:
raw_train_dataset = raw_datasets["train"]
#print(f"train data set is", raw_train_dataset[0])

raw_test_dataset = raw_datasets["test"]
#print(f"test dataset is", raw_test_dataset[0])

'''
label is of type ClassLabel, and the mapping of integers to label name is stored in the names folder.
0 corresponds to not_equivalent, and 1 corresponds to equivalent
'''
#print(f"features of train dataset", raw_train_dataset.features)

'''
To keep the data as a dataset, we will use the Dataset.map() method. This also allows us some extra 
flexibility, if we need more preprocessing done than just tokenization. 
The map() method works by applying a function on each element of the dataset
'''

from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

'''
tokenize_function returns a dictionary with the keys input_ids, attention_mask, and token_type_ids, 
so those three fields are added to all splits of our dataset
'''
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
print(f"tokenized data sets are :", tokenized_datasets)
'''
tokenized data sets are : DatasetDict({
    train: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 408
    })
    test: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 1725
    })
})
'''

'''
The last thing we will need to do is pad all the examples to the length of the longest element 
when we batch elements together — a technique we refer to as dynamic padding.

we have to define a collate function that will apply the correct amount of padding to the items of 
the dataset  we want to batch together

 let’s grab a few samples from our training set that we would like to batch together. 
 Here, we remove the columns idx, sentence1, and sentence2 as they won’t be needed and contain strings
'''
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
[len(x) for x in samples["input_ids"]]

batch = data_collator(samples)
print(f"print the batch:",{k: v.shape for k, v in batch.items()})

'''
Training:
The first step before we can define our Trainer is to define a TrainingArguments class that will contain 
all the hyperparameters the Trainer will use for training and evaluation. 
The only argument you have to provide is a directory where the trained model will be saved, 
as well as the checkpoints along the way
'''
from transformers import TrainingArguments, AutoModelForSequenceClassification, Trainer
training_args = TrainingArguments("test-trainer", eval_strategy="epoch")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

'''
valuation
Let’s see how we can build a useful compute_metrics() function 
and use it the next time we train. The function must take an EvalPrediction object
(which is a named tuple with a predictions field and a label_ids field) and will return a dictionary 
mapping strings to floats (the strings being the names of the metrics returned, and the floats their values).
To get some predictions from our model, we can use the Trainer.predict() command:
'''
import evaluate
import numpy as np

def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

'''
define a Trainer by passing it all the objects constructed up to now — the model, the training_args, 
the training and validation datasets, our data_collator, and our tokenizer:
'''
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# To fine-tune the model on our dataset, we just have to call the train() method of our Trainer:
trainer.train()

predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)
