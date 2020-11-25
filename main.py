import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import numpy as np
import torch.utils.data as dt
import torchvision.models as models

#device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#Import dataset
lfw_dataset = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

#Hyper parameters
input_size = 50*37 # 50x37 images
hidden_size = 100
num_classes = lfw_dataset.target_names.shape[0] # 10 classes in MNIST

num_epochs = 2
batch_size = 64
learning_rate = 0.001

#Transform
transform = torchvision.transforms.Compose([transforms.CenterCrop(36)])

#LFW
X = lfw_dataset.images
y = lfw_dataset.target
print(X.shape)
X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size=0.25, random_state=42)

train_dataset = dt.TensorDataset(torch.from_numpy(X_train),  torch.from_numpy(y_train))
test_dataset = dt.TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test))

train_loader = dt.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_loader = dt.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)

# examples = iter(train_loader)
# samples, labels = examples.next()
# print(samples.shape, labels.shape)


class NeuralNet(nn.Module):
    def __init__(self, num_classes):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(128*5*5, 220) # Why ?
        self.fc2 = nn.Linear(220, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.pool(F.relu(self.conv1(x)))
        out = self.pool(F.relu(self.conv2(out)))
        print(out.shape)
        out = out.view(-1, 128*5*5)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return(out)

model = models.resnet18(pretrained = True, num_classes=1000)
model.fc = nn.Linear(model.fc.in_features, num_classes)

#Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

#training loop
n_total_steps = len(train_loader)
for epochs in range(num_epochs):
    for i, (img, labels) in enumerate(train_loader):
        # print(f'i : {i}, label : {labels}')
        # 63, 50 , 37
        # Input Size is 50*37
        # 64, 50*37
        img = img.unsqueeze(1)
        img = img.expand(-1, 3, -1, -1)
        # print(img.shape)
        img = img.to(device)
        labels = labels.to(device)

        #Forward pass
        outputs = model(img)
        loss = criterion(outputs, labels)

        #Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%5 == 0:
            print(f'epochs {epochs+1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')



# Test and evaluation :

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader :
        images = images.unsqueeze(1)
        images = images.expand(-1, 3, -1, -1)
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        #Value, index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

    acc = 100 * n_correct / n_samples
    print(f' Accuracy = {acc}')