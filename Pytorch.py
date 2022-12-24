# importing the necessary modules 

import torch 
from torch import nn 
from torch.utils.data import DataLoader 
from torchvision import datasets 
from torchvision.transforms  import ToTensor 
from torch import optim
import time
# downloading the dataset 

training_data = datasets.FashionMNIST(
    root='data',
    train = True , 
    download= True ,
    transform=ToTensor()
)

testing_data = datasets.FashionMNIST(
    root='data',
    train = False , 
    download= True ,
    transform=ToTensor()
)


# Create a DataLoader to batching the dataset for parallel processing 
batch_size = 64
train_loader = DataLoader(training_data , batch_size= batch_size)
test_loader = DataLoader(testing_data , batch_size= batch_size)

# set the training device to GPU 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# let's Define the architecture of the network 
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_rule_stack = nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    def forward(self, x ):
        x = self.flatten(x)
        logits = self.linear_rule_stack(x)
        return logits

model = NeuralNetwork().to(device)

# defining loss function and optimizer to optimize the model parameters 
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters() , lr=1e-3)


# train the model 

def train(dataloader, model , loss_fn , optimizer):
    size = len(dataloader.dataset)
    model.train() 
    for batch , (X,y) in enumerate(dataloader):

        # put the input data into GPU
        X,y = X.to(device) ,y.to(device)

        # computing the prediction error
        pred = model(X)
        loss = loss_fn(pred , y)

        # Backpropagation 
        optimizer.zero_grad() 
        '''
        By calling optimizer.zero_grad(), you reset the gradients to zero,
        ensuring that the gradients from the previous iteration do not 
        affect the current iteration.
        '''
        loss.backward()
        optimizer.step()
        if batch % 100 == 0 :
            loss, current = loss.item() , batch*len(X)
            print(f'loss: {loss:>7f} [{current:>5d} /{size:>5d} ]')

# evaluated the model on test data 
def test(dataloader , model , loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss , correct = 0,0 
    with torch.no_grad():
        for X , y in dataloader:
            X ,y = X.to(device) , y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred , y ).item()
            correct += (pred.argmax(1) ==  y).type(torch.float).sum().item()
    test_loss /=num_batches
    correct /= size 
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



epochs = 5 
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model , loss_fn , optimizer)
    test( test_loader , model , loss_fn)
print('Done!!')

# save the model parameters 
torch.save(model.state_dict() ,'model.pth')
print("Saved the model state in model.pth file ")


# load the saved model parameters 
model = NeuralNetwork()
model.load_state_dict(torch.load('model.pth'))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x , y = testing_data[22][0] , testing_data[22][1]
with torch.no_grad():
    pred = model(x)
    predicted , actual = classes[pred[0].argmax(0)] , classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')