
"""
NGI internal Machine Learning course.
Session 10.06.2022
Topic: Image classification of rocktypes

Dataset:
https://www.kaggle.com/datasets/neelgajare/rocks-dataset

TODO: Try to classify better with:
- better tuning of parameters
- weighting in loss function
- different learning rate in backbone
- other backbones
- Pytorch Lightning
- Better augmentation techniques. Cropping?
- Changes in dataset. Remove obvious crazy images.

@author: Tom F. Hansen, Georg H. Erharter 
"""
# IMPORTS
######################################################################################

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
from torch.optim.optimizer import Optimizer
from torchvision import datasets, models, transforms

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import pickle
from rich.traceback import install
from rich.progress import track
from typing import Tuple

# SETUPS
######################################################################################

# presenting better error messages using rich
install()

ROOT = Path.cwd()
DATA_DIR = Path(
    "/mnt/c/Users/TFH/NGI/TG Machine Learning - General/2022 ML workshop series/datasets/Rocks")
TEST_SIZE = 0.3
NUM_WORKERS = 12
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 64
LR = 0.01
MOMENTUM = 0.9
STEP_SIZE = 3 #decay lr every xxx epoch
GAMMA = 0.1 #decay factor for multiplication
NUM_EPOCHS = 10

# stop randomness for model comparison
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


# DEFINE IMAGE AND LABEL PROVIDER
######################################################################################
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
# due to lazy_loading I don't allocate more memory of using the same dataset here
train_dataset = datasets.ImageFolder(root=DATA_DIR, transform=data_transforms["train"])
test_dataset = datasets.ImageFolder(root=DATA_DIR, transform=data_transforms["test"])

# create splitting indices for samples - could also use SubsetRandomSampler
num_classes = len(train_dataset.classes)
indices = np.arange(len(train_dataset))
labels = train_dataset.targets
train_ind, test_ind, _, _ = train_test_split(
    indices,labels, test_size=TEST_SIZE, stratify=labels)

train_set = Subset(train_dataset,indices=train_ind)
test_set = Subset(test_dataset,indices=test_ind)

train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,num_workers=NUM_WORKERS)
test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,num_workers=NUM_WORKERS)

# sample = iter(train_dataloader).next()
# sample[0] # batch of images
# sample[1] # batch of labels
# image1 = sample[0][0]
# image1.shape


# METHODS FOR TRAINING AND EVALUATION
######################################################################################
def train_epoch(
    device, 
    model: nn.Module, 
    optimizer: Optimizer, 
    loss_function: nn.CrossEntropyLoss, 
    dataloader: DataLoader, 
    ):
    """Train model for all samples in one epoch.
    Returning loss, labels, predictions"""
    
    epoch_loss = []
    epoch_labels = np.array(())
    epoch_preds = np.array(())
    
    # looping over all batches of samples
    for images, labels in track(dataloader,description="Training batches: "):
        images = images.to(device)
        labels = labels.to(device)
        
        logits = model(images) # forward pass
        _, predictions = torch.max(logits, 1) # return class label from max logit
        loss: torch.Tensor = loss_function(logits, labels)
        
        optimizer.zero_grad() #zero gradients
        loss.backward() #calculates new weights using backward propagation
        optimizer.step() #make one learning step in stocastic gradient descent
        
        # reporting - make sure no operations affect computational graph
        with torch.no_grad():
            epoch_loss.append(loss.item())
            epoch_labels = np.concatenate((epoch_labels, labels.cpu().numpy()))
            epoch_preds = np.concatenate((epoch_preds, predictions.cpu().numpy()))
        
    return np.mean(epoch_loss), epoch_labels, epoch_preds

def test_epoch(
    device, 
    model: nn.Module, 
    loss_function: nn.CrossEntropyLoss, 
    dataloader: DataLoader, 
    )->Tuple[float, list, list]:
    """Test model for all samples in one epoch.
    Returning loss, labels, predictions"""
    
    epoch_loss = []
    epoch_labels = np.array(())
    epoch_preds = np.array(())
    
    # looping over all batches of samples
    for images, labels in track(dataloader,description="Testing batches: "):
        images = images.to(device)
        labels = labels.to(device)
        
        logits = model(images) # forward pass
        _, predictions = torch.max(logits, 1) # return class label from max logit
        loss: torch.Tensor = loss_function(logits, labels)
        
        # reporting - make sure no operations affect computational graph
        with torch.no_grad():
            epoch_loss.append(loss.item())
            epoch_labels = np.concatenate((epoch_labels, labels.cpu().numpy()))
            epoch_preds = np.concatenate((epoch_preds, predictions.cpu().numpy()))
        
    return np.mean(epoch_loss), epoch_labels, epoch_preds


# DEFINE NETWORK, LOSSFUNCTION, OPTIMIZER, LR-SCHEDULER
######################################################################################
print(f"cuda is available: {torch.cuda.is_available()}. Device is {DEVICE}")
model = models.resnet18(pretrained=True) #pretrained on 1000-class Imagenet database
#turn of gradient update (learning) in backbone model
for param in model.parameters():
    param.requires_grad = False

# Parameters in newly added layers have gradient update set to True by default
num_featuremap = model.fc.in_features # num input to last fully connected layer
model.fc = nn.Linear(in_features=num_featuremap, out_features=num_classes)

model.to(DEVICE)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.fc.parameters(), lr=LR, momentum=MOMENTUM)
LR_scheduler = lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)


# TRAINING, EVALUATION, METRICS
######################################################################################

performance = []
performance_path = Path("Reports/rock_classification_performance.pkl")
tensorboard_path = Path("Reports/tensorboard_logs")
writer = SummaryWriter(log_dir=tensorboard_path)

for epoch in range(NUM_EPOCHS):
    # train
    model.train() # sets model in training mode. Turns on gradient update
    loss_train, train_labels, train_predictions = train_epoch(DEVICE,model, optimizer, loss_function,train_dataloader)
    # acc_train = accuracy(train_predictions, train_labels, average="macro") # macro is balanced accuracy
    acc_train = balanced_accuracy_score(train_labels, train_predictions)
    
    # test
    model.eval() # Freeze model weights. No model update
    loss_test, labels, predictions = test_epoch(DEVICE, model, loss_function, test_dataloader)
    acc_test = balanced_accuracy_score(labels, predictions)
    
    # report metrics
    print(f"Train-loss: {loss_train:.3f}. Train-acc: {acc_train:.2f}. \
        Test-loss: {loss_test:.3f}. Test-acc: {acc_test:.3f}")

    # add data to Tensorboard for live reporting
    writer.add_scalars("Loss development",{
        "Loss train": loss_train,
        "Loss test": loss_test
    }, global_step=epoch)
    writer.add_scalars("Accuracy development",{
        "Accuracy train": acc_train,
        "Accuracy test": acc_test
    }, global_step=epoch)

    # save data to pickle file every epoch. Load data for later analysis
    performance.append({ 
    'epoch': epoch + 1,  #epoch counts from 0
    'train_loss': loss_train,
    'train_acc': acc_train,
    'test_loss': loss_test,
    'test_acc': acc_test,
    'test_labels':labels,
    'test_predictions':predictions
    })
    pickle.dump(performance, open(performance_path, 'wb'))

writer.close()
# save trained model for later predictions and analysis
torch.save(model, Path("Reports/rock_model.pth"))