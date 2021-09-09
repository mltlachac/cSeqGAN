#!/usr/bin/env python
# coding: utf-8

# In[62]:


import sys
import os
import re
import pandas as pd 
import jupyter
import ipywidgets 
import torch
import torch.nn.functional as F
import transformers
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from transformers import  BertModel, AutoTokenizer, AutoModelForSequenceClassification


# In[64]:
argIn = sys.argv[1]
argOut = sys.argv[2]
randSeed = sys.argv[3]
df = pd.read_csv(argIn, header = 0)

# In[22]:


df = df.dropna()

#print(df)
#print(df["sentiment"])

for index, entry in df.iterrows():
    wordNum = 0
    for element in entry["text"]:
        if element == " ":
            wordNum = wordNum + 1
    if wordNum <= 1:
        df = df.drop(df.loc[df.index==index].index)
        
#print(df)
            
df = df.groupby("sentiment").sample(n = 1500, random_state = int(randSeed), replace=False)

df["depressed"] = df["sentiment"] == 1
df["depressed"].value_counts()

texts = df["text"].tolist()
temp = df["depressed"].tolist()
punc = "/!@#$%^&*[]{}()|?<>.,'''``:;-_"

for i in range(len(texts)):
    for j in texts[i]:
        if j in punc:
            texts[i] = texts[i].replace(j,"")
    texts[i] = texts[i].lower()
    
for i in range(len(texts)):
    newEnt = ""
    oldEnt = texts[i].split()
    for j in oldEnt:
        if not(j == "") and not(newEnt == ""):
            newEnt = newEnt + " " + j
        elif not(j == ""):
            newEnt = newEnt + j
    texts[i] = newEnt
#print(texts)


labels = []
for label in temp:
    if label == True:
        labels.append(float(0))
    else:
        labels.append(float(1))

        

#split data into training and testing with 20% of data being used for testing
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=.2)

#bring in the BERT model and the corresponding tokenizer
#TODO: check if more preprocessing needs to be done
#bertModel = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=1)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

#these return BatchEncoding objects
#docs for this is found at: https://huggingface.co/transformers/internal/tokenization_utils.html#transformers.tokenization_utils_base.PreTrainedTokenizerBase.from_pretrained
#look under the "__call__" heading
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)


# In[23]:


#print(train_labels.count(0.0))
#print(train_labels.count(1.0))
#print(test_labels.count(0.0))
#print(test_labels.count(1.0))


# In[24]:


class textMessageDataset(torch.utils.data.Dataset):
    
    #initializes the object (create global vars from the params)
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    #for all of the input text strings (called encodings here) create tensors (similar to arrays) and 
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

#a function that returns one if inVal > .5 and 0 if not
def get_pred(inVal): 
    if inVal > .5: 
        return float(1)
    else: 
        return float(0)


# In[25]:


trainDataset = textMessageDataset(train_encodings, train_labels)
testDataset = textMessageDataset(test_encodings, test_labels)


# In[26]:


def countNonZero(inList): 
    return len([item for item in inList if item != 0])

inputLens = [countNonZero(item) for item in train_encodings["input_ids"]]
#plt.hist(inputLens)
len(train_encodings["input_ids"])


# In[27]:


#define our loss function as binary cross entropy 
def lossFunc(inputs, targets):
   return F.binary_cross_entropy(inputs,targets)

#make a classification based on a val between 0 and 1 
def classify(inVal): 
    if inVal > .5: 
        return 1
    return 0


# In[28]:


#CREATE THE MODEL 


class DepClassifier(nn.Module):
    
  #this initializes the layers of the model 
  def __init__(self):
    print("working")
    super(DepClassifier, self).__init__()
    
    self.bert = BertModel.from_pretrained("bert-base-uncased")
    
    #NOTE: dropout rate. Higher gives more regularization and can be used to fix overfitting 
    #too high and it won't learn 
    self.drop = nn.Dropout(p=0.3)
    
    self.l1 = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
    
    self.l2 = nn.Linear(self.bert.config.hidden_size, 1)
  
  #what the data goes through to produce the output from the model 
  def forward(self, input_ids, attention_mask):
    
    bertOutput = self.bert(

      input_ids=input_ids,

      attention_mask=attention_mask
    )
    #output from the BERT portion 
    pooled_output = bertOutput.pooler_output
    
    #dropout layer 
    x = self.drop(pooled_output)
    
    #fully connected layer 
    x = self.l1(x)
    x = F.relu(x)
    x = self.drop(x)
    

    x = self.l2(x)
    
    #apply sigmoid function and return 
    return F.sigmoid(x)


# In[58]:



def getValidationStats(inModel): 
    test_loader = DataLoader(testDataset, batch_size=24, shuffle=True)

    validGroupedOutputs = torch.tensor([]).to(device)
    validGroupedLabels = torch.tensor([]).to(device)

    i = 0 
    # Testing loop
    with torch.no_grad():
        
        # Iterate over the test data and generate predictions
        for batch in test_loader:
            i += 1
            
            # Get inputs
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Generate outputs
            outputs = inModel(input_ids, attention_mask=attention_mask)

            #concat new labels to larger list of labels   
            validGroupedOutputs = torch.cat((validGroupedOutputs, outputs), 0)
            validGroupedLabels = torch.cat((validGroupedLabels, labels), 0)
        
        #get the accuraccy across all of the validation set 
        #double check that sigmoid doesn't need to be applied here 
        outputArr = torch.flatten(validGroupedOutputs).detach().to("cpu").numpy()
        predictions = list(map(classify, outputArr))
        #print(predictions)
        #print(validGroupedLabels.to("cpu"))
        validDf = pd.DataFrame({"predictions":predictions , "GT":list(validGroupedLabels.to("cpu"))})
        print(validDf.head(50))
        CM = confusion_matrix(validGroupedLabels.to("cpu").numpy(), predictions)
        TP_rate = (CM[1][1])/(CM[1][1] + CM[0][1])
        TN_rate = (CM[0][0])/(CM[1][0] + CM[0][0])
        Acc = (CM[1][1] + CM[0][0])/(CM[1][1] + CM[0][0] + CM[1][0] + CM[0][1])

        return [TP_rate, TN_rate, Acc, CM]


# In[59]:


def getTrainSS(outputs, labels): 
    outputsArr = torch.flatten(outputs).detach().to("cpu").numpy()
    outputsArr = list(map(classify, outputsArr))
    labelsArr = labels.to("cpu").numpy()
    #print("Train Guesses: \n" + str(outputsArr) + "\nTrain Labels: \n" + str(labelsArr) + "\n\n")
    return accuracy_score(labelsArr, outputsArr)


# In[60]:


import gc

gc.collect()

torch.cuda.empty_cache()

from torch.utils.data import DataLoader
from transformers import AdamW

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = DepClassifier()
model.to(device)

#put the model in training mode

#train loader breaks up the input data into batchs according to batch size
#NOTE: change batch size here 
train_loader = DataLoader(trainDataset, batch_size=32, shuffle=True)

#NOTE: optimizer determines how quickly/how the model finds the best weights 
#NOTE: adjust learning rate here, eps doesn't make much difference 
optim = AdamW(model.parameters(), lr=2e-5, eps=2e-8)

# from https://stackoverflow.com/questions/52660985/pytorch-how-to-get-learning-rate-during-training#:~:text=For%20only%20one%20parameter%20group%20like%20in%20the,and%20simply%20call%20the%20built-in%20lr_scheduler.get_lr%20%28%29%20method.
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
groupedOutputs = torch.tensor([]).to(device)
groupedLabels = torch.tensor([]).to(device)

trainMetrics = []
validationMetrics = []

#look up the great tutorial by sentdex for help/explanation 
#https://www.youtube.com/watch?v=9j-_dOze4IM
i = 0 

#NOTE: Begin training loop 
#NOTE: change epoch_num to cahnge number of epochs 
EPOCH_NUM = 4
for epoch in range(EPOCH_NUM):
    print("EPOCH " + str(epoch) + "\n\n\n\n\n")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache() 
        
    for batch in train_loader:
        
        #before passing new data we have to set the gradient to zero again 
        optim.zero_grad()

        #this is just telling pytorch that these tensors can be processed on our device
        #this device is going to be a gpu in our case 
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask)
        
        groupedOutputs = torch.cat((groupedOutputs, outputs), 0)
        groupedLabels= torch.cat((groupedLabels, labels), 0)
        
        #NOTE: this is how we get the loss which determines how the model gets updated 
        loss = lossFunc(outputs, labels.unsqueeze(1))
        
        #this if statement doesn't train the model, just evaluates. Look into functions in this section to compute different metrics
        if i % 30 == 0: 
            model.eval()
            groupedLoss = lossFunc(groupedOutputs, groupedLabels.unsqueeze(1))
            trainAcc = getTrainSS(groupedOutputs, groupedLabels)
            trainMetrics.append([i, groupedLoss, trainAcc])
            print("Train Metrics:\nAccuracy - " + str(trainAcc) + "\n\n")
            
            TPR_TNR = getValidationStats(model)
            validationMetrics.append(TPR_TNR)
            print("Validation Metrics:\nTrue Positive Rate - " + str(TPR_TNR[0]) + "\nTrue Negative Rate - " + str(TPR_TNR[1]) + "\n\n")
            
            #redefine so that we don't just keep adding on to these 
            groupedOutputs = torch.tensor([]).to(device)
            groupedLabels = torch.tensor([]).to(device)
            model.train()
        #find the gradients for all of the nodes 
        loss.backward()
        
        #take a "step" by changing the weights according to the learning rate 
        optim.step()
        
        i += 1


# In[66]:


outputDf = pd.DataFrame({"trainMetrics":trainMetrics, "validMetrics":validationMetrics})
MY_OUTPUT_PATH = argOut
outputDf.to_csv(MY_OUTPUT_PATH)


# In[57]:


print(trainMetrics)
print(validationMetrics)


# In[ ]:




