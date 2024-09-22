'''
Architecture: This is the skeleton of the model — the definition of each layer and each operation that happens
within the model.
Checkpoints: These are the weights that will be loaded in a given architecture.
Model: This is an umbrella term that is not as precise as “architecture” or “checkpoint”: 
it can mean both. This course will specify architecture or checkpoint when it matters to reduce ambiguity.
refer doc:
https://huggingface.co/learn/nlp-course/chapter1/5?fw=pt
https://huggingface.co/learn/nlp-course/chapter2/2?fw=pt

'''

import torch
from transformers import  pipeline, AutoModel, AutoTokenizer, AutoModelForSequenceClassification


classifier = pipeline("sentiment-analysis")
classifier(
    [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
)


"""
Flow:  
------> Tokenizer --------> Model ----------> Post Processing
Raw text -----> Input Ids ------> Logits ------->  Predictions
"""

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)

"""
Batch size: The number of sequences processed at a time (2 in our example).
Sequence length: The length of the numerical representation of the sequence (16 in our example).
Hidden size: The vector dimension of each model input.
"""

model = AutoModel.from_pretrained(checkpoint)
automodeloutputs = model(**inputs)
print(automodeloutputs.last_hidden_state.shape) # torch.Size([2, 16, 768])

"""
we will need a model with a sequence classification head 
(to be able to classify the sentences as positive or negative)
"""

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
print(outputs.logits.shape)# torch.Size([2, 2]) two sentences and two labels, the result we get from our model is of shape 2 x 2.

print(outputs.logits)

"""
To be converted to probabilities, they need to go through a SoftMax layer 
(all 🤗 Transformers models output the logits, as the loss function for training will generally fuse the last
activation function, such as SoftMax, with the actual loss function, such as cross entropy):
"""
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)

# Get the labels
print(model.config.id2label)
 