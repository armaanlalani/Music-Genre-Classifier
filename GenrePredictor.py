# Used in Google Colab
'''
from google.colab import drive
drive.mount('/content/gdrive')
%cd gdrive
%cd MyDrive
%cd ECE324\:\ Project
%cd Code
'''

%matplotlib inline
import numpy as np 
import torch 
from torch.autograd import Variable 
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image 
import torchvision
import matplotlib.pyplot as plt 
from torch.optim import SGD,Adam 
import torch.nn as nn 
import time as time 
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F

class CNNBaseline(nn.Module): # the baseline cnn model
    def __init__(self):
        super(CNNBaseline,self).__init__()
        self.conv1 = nn.Conv2d(3,10,3)
        self.conv2 = nn.Conv2d(10,10,3)
        self.conv3 = nn.Conv2d(10,10,3)
        self.conv4 = nn.Conv2d(10,10,3)
        self.pool = nn.MaxPool2d(2,2)
        self.bn = nn.BatchNorm2d(10)
        self.fc1 = nn.Linear(6090,1000)
        self.fc2 = nn.Linear(1000,200)
        self.fc3 = nn.Linear(200,7)
    def forward(self,x):
        x = self.pool(F.sigmoid(self.bn(self.conv1(x))))
        x = self.pool(F.sigmoid(self.bn(self.conv2(x))))
        x = self.pool(F.sigmoid(self.bn(self.conv3(x))))
        x = self.pool(F.sigmoid(self.bn(self.conv4(x))))
        x = x.view(-1,6090)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN(nn.Module): # cnn model
  def __init__(self):
    super(CNN,self).__init__()
    self.conv1 = nn.Conv2d(3,16,3)
    self.pool = nn.MaxPool2d(2,2)
    self.conv2 = nn.Conv2d(16,32,3)
    self.conv3 = nn.Conv2d(32,32,3)
    self.conv4 = nn.Conv2d(32,32,3)
    self.bnc1 = nn.BatchNorm2d(16)
    self.bnc2 = nn.BatchNorm2d(32)
    self.fc1 = nn.Linear(19488,4000)
    self.fc2 = nn.Linear(4000,500)
    self.fc3 = nn.Linear(500,8)
  def forward(self,x):
    x = self.pool(F.relu(self.bnc1(self.conv1(x))))
    x = self.pool(F.relu(self.bnc2(self.conv2(x))))
    x = self.pool(F.relu(self.bnc2(self.conv3(x))))
    x = self.pool(F.relu(self.bnc2(self.conv4(x))))
    x = x.view(-1,19488)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

class CRNN(nn.Module): # crnn model
  def __init__(self):
    super(CRNN,self).__init__()
    self.conv1 = nn.Conv2d(3,16,3)
    self.pool = nn.MaxPool2d(2,2,padding=1)
    self.conv2 = nn.Conv2d(16,32,3)
    self.conv3 = nn.Conv2d(32,32,3)
    self.conv4 = nn.Conv2d(32,32,3)
    self.bnc1 = nn.BatchNorm2d(16)
    self.bnc2 = nn.BatchNorm2d(32)
    self.gru = nn.GRU(input_size=23, hidden_size=500)
    self.fc1 = nn.Linear(500,7)
  def forward(self,x):
    x = self.pool(F.relu(self.bnc1(self.conv1(x))))
    x = self.pool(F.relu(self.bnc2(self.conv2(x))))
    x = self.pool(F.relu(self.bnc2(self.conv3(x))))
    x = self.pool(F.relu(self.bnc2(self.conv4(x))))

    x = torch.reshape(x,(-1,23,31*32))
    x = torch.transpose(x,1,2)
    x, states = self.gru(x)
    x = x.mean(1)

    x = self.fc1(x)
    return x

def getStats(path):
    """
        Input:
        path --> Path to Image Folder

        Returns:
        avg --> mean of channels
        std --> stdDev of channels
        transform --> transform of images/tensors
    """
    # temporary ds to find mean and std
    tempDS = ImageFolder(path)

    # find avg and std in ds
    avg = [0,0,0]; std = [0,0,0]
    for pic in tempDS:
        # PIL needs pixel value b/w 0 and 1.
        img = np.asarray(pic[0])/255
        for i in range(3):
            avg[i] += (img[:,:,i].mean())/len(tempDS)
            std[i] += ((img[:,:,i]).std())/(len(tempDS))
    print(f'AVGS: {avg} \nSTDS: {std}')
    # transform each image to a tensor and normalize
    transform = transforms.Compose(
    [transforms.ToTensor(),transforms.Normalize(avg,std)]
    )
    return avg,std,transform

def loadData(batchSize,transform,shuffle=True,seed=0,split = {'train': 0.7, 'valid': 0.2, 'test': 0.1}):
    train = ImageFolder("./Dataset",transform=transform)
    size = len(train)
    idxs = list(range(size))

    print(f"train.classes: \n{train.classes}")

    # holds the indexes of the respective classes
    country = idxs[:499]
    edm = idxs[499:499*2]
    hiphop = idxs[499*2:499*3]
    latin = idxs[499*3:499*4]
    pop = idxs[499*4:499*5] 
    rb = idxs[499*5:499*6] 
    rock = idxs[499*6:]

    # randomly shuffles the index within each class
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(country)
        np.random.shuffle(edm)
        np.random.shuffle(hiphop)
        np.random.shuffle(latin)
        np.random.shuffle(pop)
        np.random.shuffle(rb)
        np.random.shuffle(rock)

    # stopping indexes for training and validation
    trainStop = int(np.floor(
        499 * (split['train'])
    ))

    validStop = int(np.floor(
        trainStop + (499 * (split['valid']))
    ))

    idxs = [country, edm, hiphop, latin, pop, rb, rock]

    # adds the indexes for training, validation, and testing to the appropriate lists
    train_idxs = []
    valid_idxs = []
    test_idxs = []
    for i in range(7):
        for j in range(trainStop):
            train_idxs.append(idxs[i][j])
        for j in range(trainStop, validStop):
            valid_idxs.append(idxs[i][j])
        for j in range(validStop,499):
            test_idxs.append(idxs[i][j])

    # creates the dataloaders based on the indices
    splitIdxs = {'train': train_idxs, 'valid': valid_idxs, 'test': test_idxs}
    loaders = {"trainLoader": None,"validLoader": None,"testLoader": None}

    for split in splitIdxs:
        for loader in loaders:
            if (split + "Loader" == loader) and loaders[loader] == None:
                sampler = SubsetRandomSampler(splitIdxs[split])
                loaders[loader] = DataLoader(train,batch_size=batchSize,sampler=sampler)

    return loaders

# used for the training dataset
def evaluate(outputs, labels, loss_fnc):
  output = outputs.detach().numpy()
  label = labels.detach().numpy()
  count = 0
  # determines the number of correct predictions and returns the accuracy
  for i in range(output.shape[0]):
    if np.argmax(output[i]) == label[i]:
      count += 1
  return count / output.shape[0]

# used for the validation dataset
def evaluate1(model, val_loader, batch_size, loss, val_loss, loss_fnc):
  eval, count, total_loss = 0, 0, 0
  for i, data in enumerate(val_loader):
    inputs, labels = data
    inputs = inputs.float()
    labels = labels.float()
    outputs = model(inputs) # determines the outputs based on the current model
    outputs = outputs.type(torch.float)
    labels = labels.type(torch.LongTensor)
    eval += evaluate(outputs, labels, loss_fnc) # uses the evaluate function to determine the accuracy
    count += 1

    loss_in = loss(input=outputs, target=labels)
    total_loss += float(loss_in.item())
  val_loss.append(total_loss/count) # updates the validation loss list
  return eval/count # returns the overall accuracy of the entire validation dataset

torch.manual_seed(0)

lr = 0.001
epochs = 75

eval_every = 77

model = CRNN()
loss_fnc = torch.nn.CrossEntropyLoss()
opt = Adam(model.parameters(),lr)
# opt = SGD(model.parameters(),lr)

batch_size = 32

def main():
    sets = loadData(batch_size,transform)
    train,valid,test = sets['trainLoader'], sets['validLoader'], sets['testLoader'] # creates the train/validation/test dataloaders
    valid_acc, train_acc, valid_loss, train_loss, time_total = [], [], [], [], []
    first_time = time.time() # starts the timer

    classes = [0,0,0,0,0,0,0] # verifies equal class representation in the test dataset
    for j, data in enumerate(test):
      inputs, labels = data
      inputs = inputs.float()
      labels = labels.float()
      for i in labels.detach().numpy():
          classes[int(i)] += 1

    print('Test:')
    print(classes)

    classes = [0,0,0,0,0,0,0] # verifies equal class representation in the training dataset
    for j, data in enumerate(train):
      inputs, labels = data
      inputs = inputs.float()
      labels = labels.float()
      for i in labels.detach().numpy():
          classes[int(i)] += 1

    print('Training:')
    print(classes)

    classes = [0,0,0,0,0,0,0] # verifies equal class representation in the validation dataset
    for j, data in enumerate(valid):
      inputs, labels = data
      inputs = inputs.float()
      labels = labels.float()
      for i in labels.detach().numpy():
          classes[int(i)] += 1

    print('Validation:')
    print(classes)

    for epoch in range(0,epochs,1):
        total_loss, total_corr, train_eval = 0, 0, 0
        for i, data in enumerate(train, 0):
            print(i) # prints the batch number
            inputs, labels = data
            inputs = inputs.float()
            labels = labels.float()

            opt.zero_grad()

            outputs = model(inputs) # obtains the outputs of the batch

            outputs = outputs.type(torch.float)
            labels = labels.type(torch.LongTensor)

            loss_in = loss_fnc(input=outputs, target=labels)
            loss_in.backward() # computes the loss
            opt.step()

            total_loss = float(total_loss + loss_in.item()) # updates the overall loss
            train_eval = train_eval + evaluate(outputs, labels, loss_fnc) # updates the training accuracy

            if i % eval_every == eval_every-1: # number of evaluations per epoch --> set at 77 to ensure only once for epoch

                acc = evaluate1(model, valid, batch_size, loss_fnc, valid_loss, loss_fnc) # determines the validation accuracy
                print('Validation Accuracy: ' + str(acc))
                valid_acc.append(acc)
                train_acc.append(train_eval / ((i%eval_every)+1)) # determines the training accuracy
                print('Training Accuracy: ' + str(train_eval / ((i%eval_every)+1)))
                time_diff = time.time() - first_time # determines the elapsed time
                print('  Elapsed Time: ' + str(time_diff) + ' seconds')
                train_loss.append(total_loss / ((i%eval_every)+1))
                train_eval = 0
                print('[%d, %5d] loss: %.5f' %
                    (epoch + 1, i + 1, total_loss / ((i%eval_every)+1))) # prints the overall loss
                total_loss = 0

    
    test_loss = 0
    test_acc = 0
    true = [] # used for confusion matrix
    pred = [] # used for confusion matrix
    for j, data in enumerate(test):
      inputs, labels = data
      inputs = inputs.float()
      labels = labels.float()
      for i in labels.detach().numpy():
          true.append(i) # obtains the true label values
      outputs = model(inputs) # determines the outputs based on the final model
      for i in outputs.detach().numpy():
          pred.append(np.argmax(i)) # obtains the predicted values
      outputs = outputs.type(torch.float)
      labels = labels.type(torch.LongTensor)
      loss_in = loss_fnc(input=outputs, target=labels) # determines the training loss
      loss_in.backward()
      test_loss += float(test_loss + loss_in.item())
      test_acc += evaluate(outputs, labels, loss_fnc)
    print("Test Loss: " + str(test_loss/j)) # prints the test loss and test accuracy
    print("Test Accuracy: " + str(test_acc/j))

    print(confusion_matrix(true,pred)) # determines the confusion matrix

    # plots of accuracy and loss
    plt.plot(valid_acc, label = "Validation")
    plt.plot(train_acc, label = "Training")
    plt.legend()
    plt.xlabel("Number of Steps")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Number of Steps")
    plt.show()

    plt.plot(valid_loss, label = "Validation")
    plt.plot(train_loss, label = "Training")
    plt.legend()
    plt.xlabel("Number of Steps")
    plt.ylabel("Loss")
    plt.title("Loss vs. Number of Steps")
    plt.show()

main()