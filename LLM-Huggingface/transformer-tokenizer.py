import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

"""
Tokenizers are one of the core components of the NLP pipeline. They serve one purpose: to translate text 
into data that can be processed by the model. Models can only process numbers, 
so tokenizers need to convert our text inputs to numerical data

refer HF doc: https://huggingface.co/learn/nlp-course/chapter2/4?fw=pt
"""

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)

print(tokens)

ids = tokenizer.convert_tokens_to_ids(tokens) # The conversion to input IDs is handled by the convert_tokens_to_ids() tokenizer method:

print(ids)

"""
 Decoding is going the other way around: from vocabulary indices, 
 we want to get a string. This can be done with the decode() method as follows:
"""
decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
print(decoded_string)

"""
Models expect a batch of inputs.Transformers models expect multiple sentences by default. 
But below we send only one sentence hence it fail
"""

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
#input_ids = torch.tensor(ids) # did not add dimension
# This line will fail.
#model(input_ids) # IndexError: too many indices for tensor of dimension 1

# Let’s try again and add a new dimension:
input_ids = torch.tensor([ids]) #  added dimension
print("Input IDs:", input_ids)

output = model(input_ids)
print("Logits:", output.logits)

'''
 When you’re trying to batch together two (or more) sentences, they might be of different lengths. 
 If you’ve ever worked with tensors before, you know that they need to be of rectangular shape, 
 so you won’t be able to  convert the list of input IDs into a tensor directly.
 To work around this problem, we usually pad the inputs
 
 using tokenizer.pad_token_id we can pad. but it will give different results

 '''
 
sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]] # tensor([[ 0.5803, -0.4125]], grad_fn=<AddmmBackward0>)
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id], #  [ 1.3373, -1.2163]], grad_fn=<AddmmBackward0>)
]

print(model(torch.tensor(sequence1_ids)).logits)
print(model(torch.tensor(sequence2_ids)).logits)
print(model(torch.tensor(batched_ids)).logits)

'''
 To get the same result when passing individual sentences of different lengths through the model 
 or when passing a batch use attention masks
 
 Attention masks are tensors with the exact same shape as the input IDs tensor, 
 filled with 0s and 1s: 1s indicate the corresponding tokens should be attended to, 
 and 0s indicate the corresponding tokens should not be attended to 
 (i.e., they should be ignored by the attention layers of the model).

'''

batched_ids2 = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id], # [ 0.5803, -0.4125]], grad_fn=<AddmmBackward0>)
]

attention_mask = [
    [1, 1, 1],
    [1, 1, 0],
]

outputs = model(torch.tensor(batched_ids2), attention_mask=torch.tensor(attention_mask))
print(outputs.logits)

'''
With Transformer models, there is a limit to the lengths of the sequences we can pass the models. 
Most models handle sequences of up to 512 or 1024 tokens, and will crash when asked to process 
longer sequences. There are two solutions to this problem:
1. Use a model with a longer supported sequence length: sequence = sequence[:max_sequence_length]
2. truncate the sequences that are longer than the specified max length : tokenizer(sequences, max_length=8, truncation=True)
'''
 




