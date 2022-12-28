# import the necessary module for CNN
import torch
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from torch import optim  # optimizer 
from torch import nn 
from torchvision import datasets 
from torch.utils.data import DataLoader 
from tqdm import tqdm

# import the dataset from pytorch 
training_data = datasets.MNIST( 
    root = 'data',
    train = True ,
    download = True , 
    transform = ToTensor()
)

testing_data = datasets.MNIST( 
    root = 'data',
    train = False ,
    download = True , 
    transform = ToTensor()
)

# parallel processing the dataset by passing in batches 
batch_size = 64 
train_loader = DataLoader(dataset=training_data , batch_size= batch_size , shuffle= True)
test_loader = DataLoader(dataset=testing_data , batch_size= batch_size , shuffle= True)

# set the device to GPU 
device =  'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device : {device}')


# CNN Architecture 
class CNN(nn.Module):
    def __init__(self , in_channels=1 , num_classes=10):
        super(CNN , self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels= 8 , 
            kernel_size= 3,
            stride= 1 , 
            padding= 1 
            )
        self.pool = nn.MaxPool2d(kernel_size=2 , stride= 2)
        self.conv2 = nn.Conv2d(
            in_channels=8 , 
            out_channels= 16 , 
            kernel_size= 3 , 
            stride= 1 , 
            padding=1
        )
        self.fc1 = nn.Linear(16 * 7 * 7 , num_classes)

    def forward(self , x ):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x 

model = CNN().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters() , lr=1e-3)

def train(dataloader , model ,loss_fn ,optmizer):
    size = len(dataloader.dataset)
    model.train()
    for batch , (X , y) in enumerate(dataloader):
        X , y = X.to(device) , y.to(device)

        pred = model(X)
        loss = loss_fn(pred , y )

        optimizer.zero_grad()   
        loss.backward()
        optimizer.step()
        if batch % 100 == 0 :
            loss, current = loss.item() , batch*len(X)
            print(f'loss: {loss:>7f} [{current:>5d} /{size:>5d} ]')

def test(dataloader , model , loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss ,correct = 0 , 0
    with torch.no_grad():
        for X , y in dataloader:
            X , y = X.to(device) , y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred , y).item()
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

torch.save(model.state_dict() ,'model.pth')
print("Saved the model state in model.pth file ")

