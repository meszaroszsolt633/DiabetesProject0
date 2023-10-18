from functions import dataPrepare,load
from defines import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer


train,patientTrain = load(TRAIN2_540_PATH)
test,patientTest = load(TEST2_540_PATH)
trainX, trainY, validX, validY = dataPrepare(train, test, 3,15)


sequence_length = 5
feature_dim = 1


model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


model.config.num_labels = 1
model.resize_token_embeddings(len(tokenizer))


criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)


num_epochs = 10
for epoch in range(num_epochs):
    model.train()

    for i in range(len(trainX)):
        inputs, target = trainX[i], trainY[i]

        optimizer.zero_grad()


        inputs = torch.tensor(inputs, dtype=torch.float)


        outputs = model(inputs).last_hidden_state
        regression_output = outputs[:, 0]
        loss = criterion(regression_output, torch.tensor(target).float())


        loss.backward()
        optimizer.step()

# Evaluation
model.eval()

