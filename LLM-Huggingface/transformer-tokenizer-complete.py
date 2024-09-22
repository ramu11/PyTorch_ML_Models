import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

"""
Transformers API can handle all of this for us with a high-level function that we’ll dive into here. 
When you call your tokenizer directly on the sentence, you get back inputs that are ready to pass through 
your model

Here, the model_inputs variable contains everything that’s necessary for a model to operate well. 
For DistilBERT, that includes the input IDs as well as the attention mask. 
Other models that accept additional inputs will also have those output by the tokenizer object.

refer HF doc: https://huggingface.co/learn/nlp-course/chapter2/6?fw=pt
"""

from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

model_inputs = tokenizer(sequence)

print(f"single sentence", model_inputs)

'''
It also handles multiple sequences at a time, with no change in the API:

'''
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

model_inputs = tokenizer(sequences)
print(f"multiple sequence input ids", model_inputs)

# do padding
# Will pad the sequences up to the maximum sequence length
model_inputs = tokenizer(sequences, padding="longest")
print(f"multiple sequence input ids with longest padding", model_inputs)

# Will pad the sequences up to the model max length
# (512 for BERT or DistilBERT)
model_inputs = tokenizer(sequences, padding="max_length")
print(f"multiple sequence input ids with padding upto the model max_length", model_inputs)

# Will pad the sequences up to the specified max length
model_inputs = tokenizer(sequences, padding="max_length", max_length=8)
print(f"multiple sequence input ids with specified max_length", model_inputs)


# Will truncate the sequences that are longer than the model max length
# (512 for BERT or DistilBERT)
model_inputs = tokenizer(sequences, truncation=True)

# Will truncate the sequences that are longer than the specified max length
model_inputs = tokenizer(sequences, max_length=8, truncation=True)
print(f"multiple sequence input ids with specified max_length with truncation", model_inputs)


#wrapping up
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]
tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
output = model(**tokens)
print(f"model output", output)

predictions = torch.nn.functional.softmax(output.logits, dim=-1)
print(predictions)

# Get the labels
print(model.config.id2label)

