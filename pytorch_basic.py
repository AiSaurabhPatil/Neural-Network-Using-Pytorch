# Neural network using pytorch 

# importing necessary modules 
import torch
import torchvision.datasets  as datasets 
from torch import nn # Contain all the neural network module
from torch import optim # this contain optimizer function like Adam,SGD etc 
import torch.nn.functional as F # this contain activation function 
from torch.utils.data import DataLoader # this is used to load the data into batches
from torchvision.transforms import transforms
from tqdm import tqdm


# create a structure of Neural Network using class 
class NN(nn.Module): # this class is inherting the features from nn.Module
    def __init__(self, input_size , num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50 ,num_classes)

    def forward(self, x ):
        x = F.relu(self.fc1(x)) # output of fc1 is passed through relu activation function 
        x = self.fc2(x)
        return x 

# let's use GPU power to train the network
device = torch.device('cuda')

# Hyperparameters 
input_size = 784
num_classes = 10 
learning_rate = 0.01
batch_size = 50
num_epoches = 3

# Load the dataset from pytorch inbuilt collection of datasets 
train_dataset = datasets.MNIST(
    root='dataset/',
    train = True ,
    transform=transforms.ToTensor(),
    download=True
)

test_dataset = datasets.MNIST(
    root='dataset/',
    train = False ,
    transform=transforms.ToTensor(),
    download=True
)

# send the data in a batches using DataLoader 
train_loader = DataLoader(dataset=train_dataset , batch_size= batch_size , shuffle = True)
test_loader = DataLoader(dataset=test_dataset , batch_size= batch_size , shuffle = True)


# Initiate the Neural Network 
model = NN(input_size=input_size , num_classes= num_classes).to(device=device)

# Loss and optimization process 
Loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters() , lr = learning_rate)

# Let's train the network

for epoch in range(num_epoches):
    for batch_idx , (data, targets) in enumerate(tqdm(train_loader)):
        data = data.to(device = device)
        targets = targets.to(device=device)

        # flatten the batch input 
        data = data.reshape(data.shape[0], -1 )
        scores = model(data)
        # calculating the loss of network
        loss = Loss(scores,targets)

        # backward propagation to improve the network performance 
        optimizer.zero_grad()  # always reset the gradient to 0 for each input batch
        loss.backward()

        # Optimize the backward propagation 
        optimizer.step()


def check_accuracy(loader , model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x , y in loader :
            x = x.to(device=device)
            y = y.to(device=device)

            x = x.reshape(x.shape[0],-1)

            scores = model(x)
            _, predictions  = scores.max(1)
            num_correct += (predictions == y).sum()

            # Keep track of number of samples
            num_samples += predictions.size(0)
    
    model.train()
    return num_correct / num_samples

print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")