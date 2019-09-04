# from datapre import train_1,train_2,test_1,test_2,target,field_dims
# from data_preprocess import final_train,final_test,field_dims,target
from data_preprocess import final_train_int,final_test_float,final_test_int,final_train_float,field_dims,target
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import tqdm
from torch.utils.data import DataLoader
from torchfm.model.afi import AutomaticFeatureInteractionModel
from sklearn.metrics import roc_auc_score
import pandas as pd
class Kaggle_data(torch.utils.data.Dataset):
    def __init__(self, train_1,train_2,target=None):

        self.train_1 = train_1.astype(np.int64)
        self.train_2 = train_2.astype(np.float32)
        self.target = target.astype(np.float64)

    def __getitem__(self, index):

        return self.train_1[index],self.train_2[index],self.target[index]

    def __len__(self):
        return len(self.train_1)

class Kaggle_data_pre(torch.utils.data.Dataset):
    def __init__(self, train_1,train_2):

        self.train_1 = train_1.astype(np.int64)
        self.train_2 = train_2.astype(np.float32)

    def __getitem__(self, index):

        return self.train_1[index],self.train_2[index]

    def __len__(self):
        return len(self.train_1)

dataset = Kaggle_data(final_train_int,final_train_float,target)
train_length = int(len(dataset) * 0.9)
# valid_length = int(len(dataset) * 0.1)
test_length = len(dataset) - train_length
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, (train_length, test_length))
train_data_loader = DataLoader(train_dataset, batch_size=512, num_workers=8)
test_data_loader = DataLoader(test_dataset, batch_size=1024, num_workers=8)
model = AutomaticFeatureInteractionModel(
            field_dims, embed_dim=16, num_heads=4, num_layers=4, mlp_dims=(128,128,128), dropouts=(0.5,0.5)).cuda()
# model.load_state_dict(torch.load('./model/kaggle.pth'))
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.00001, weight_decay=0.001)

auc = 0
log_interval = 100
for epoch in range(50):
    model.train()
    total_loss = 0

    for i, (train_1, train_2, target) in enumerate(tqdm.tqdm(train_data_loader, smoothing=0, mininterval=1.0)):

        train_1,train_2,target = train_1.cuda(),train_2.cuda(),target.cuda()
        y = model(train_1, train_2)
        loss = criterion(y, target.float())

        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if (i + 1) % log_interval == 0:
            print('    - loss:', total_loss / log_interval)
            total_loss = 0

    targets = []
    predicts = []
    for train_1, train_2, target in tqdm.tqdm(test_data_loader, smoothing=0, mininterval=1.0):

        train_1,train_2,target = train_1.cuda(),train_2.cuda(),target.cuda()
        y = model(train_1, train_2).cpu()
        targets.extend(target.tolist())

        predicts.extend(y.tolist())
        temp_auc = roc_auc_score(targets, predicts)
        if temp_auc > auc:
            torch.save(model.state_dict(),'./model/kaggle2.pth')
    print('epoch_{}: auc_{}:'.format(epoch,temp_auc))


model.eval()
dataset = Kaggle_data_pre(final_test_int,final_test_float)
data_loader = DataLoader(dataset,batch_size=1024, num_workers=8)
predicts = []
for train_1, train_2 in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):

    train_1,train_2 = train_1.cuda(),train_2.cuda()
    y = model(train_1, train_2)

    predicts.extend(y.tolist())

# print(predicts)
test_final = pd.read_csv('./input/sample_submission.csv',index_col='TransactionID')
test_final['isFraud'] = predicts
test_final.to_csv('./input/sample_submission4.csv')


