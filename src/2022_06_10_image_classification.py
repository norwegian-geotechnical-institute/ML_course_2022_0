
"""
NGI internal Machine Learning course.
Session 10.06.2022
Topic: Image classification of rocktypes

Dataset:
https://www.kaggle.com/datasets/neelgajare/rocks-dataset

TODO: Try to classify better with:
- Tuning of hyperparameters
- different learning rate in backbone
- other backbones
- more complicated head-network
- Pytorch Lightning or Keras implementations
- Better augmentation techniques. Cropping? Filters?
- Changes in dataset. Remove obvious crazy images.
- Cross validation and other splits

Dataset need to be structured like:
ROOT > Classname > filename.jpg

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
import torchvision

import numpy as np
import numpy.typing as npt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils import class_weight
import pickle
from rich.traceback import install
from rich.progress import track
from typing import Tuple
from utility import imshow

# SETUPS
######################################################################################

# presenting better error messages using rich
install()

DATA_DIR = Path(
    "/mnt/c/Users/TFH/NGI/TG Machine Learning - General/2022 ML workshop series/datasets/Rocks")
TEST_SIZE = 0.3
NUM_WORKERS = 12
# remember to place both model and data on the same device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SAVE_PERFORMANCE = True
SHOW_BATCH = True

# Hyperparameters
BATCH_SIZE = 64
LR = 0.01
MOMENTUM = 0.9
STEP_SIZE = 4 #decay lr every xxx epoch
GAMMA = 0.3 #decay factor for multiplication
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
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # values for transfer learning model
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

# create splitting indices for samples
num_classes = len(train_dataset.classes)
indices = np.arange(len(train_dataset))
labels = train_dataset.targets
train_ind, test_ind, _, _ = train_test_split(
    indices,labels, test_size=TEST_SIZE, stratify=labels)

train_set = Subset(train_dataset,indices=train_ind)
test_set = Subset(test_dataset,indices=test_ind)


# Testing for comparison with the ants and bees dataset. You should get accuracies over 95% on that dataset.
# Note that this dataset has just 245 images in training set and still the transfer learning model works well.
# You can just uncomment this code and it should run, after you have updated with your path to the dataset
# DATA_DIR_TRAIN = Path("/home/tfha/datasets/hymenoptera_data/train")
# DATA_DIR_VAL = Path("/home/tfha/datasets/hymenoptera_data/val")

# train_dataset = datasets.ImageFolder(root=DATA_DIR_TRAIN, transform=data_transforms["train"])
# test_dataset = datasets.ImageFolder(root=DATA_DIR_VAL, transform=data_transforms["test"])
# num_classes = len(train_dataset.classes)
# train_set = train_dataset
# test_set = test_dataset


train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,num_workers=NUM_WORKERS)
test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,num_workers=NUM_WORKERS)

#VISUALIZE A BATCH OF DATA
######################################################################################

if SHOW_BATCH:
    # Get a batch of training data
    inputs, classes = next(iter(train_dataloader))
    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[train_dataset.classes[x] for x in classes])

# METHODS FOR TRAINING AND EVALUATION
######################################################################################
def train_epoch(
    device, 
    model: nn.Module, 
    optimizer: Optimizer, 
    loss_function: nn.CrossEntropyLoss, 
    dataloader: DataLoader, 
    )->Tuple[float, npt.NDArray, npt.NDArray]:
    """Train model for all samples in one epoch.
    Returning loss, labels, predictions"""
    
    epoch_loss = []
    epoch_labels: npt.NDArray = np.array(())
    epoch_preds: npt.NDArray = np.array(())
    
    # looping over all batches of samples
    for images, labels in track(dataloader,description="Training batches: "):
        images = images.to(device) #sending data to gpu or cpu
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
    )->Tuple[float, npt.NDArray, npt.NDArray]:
    """Test model for all samples in one epoch.
    Returning loss, labels, predictions"""
    
    epoch_loss = []
    epoch_labels: npt.NDArray = np.array(())
    epoch_preds: npt.NDArray = np.array(())
    
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
model = models.resnet50(pretrained=True) #pretrained on 1000-class Imagenet database

#turn off gradient update (learning) in backbone model
for param in model.parameters():
    param.requires_grad = False

# Parameters in newly added layers have gradient update set to True by default
num_featuremap = model.fc.in_features # num input to last fully connected layer
model.fc = nn.Linear(in_features=num_featuremap, out_features=num_classes)

model.to(DEVICE)

weights = class_weight.compute_class_weight(class_weight="balanced",
                                                classes=np.unique(train_dataset.targets),
                                                y=train_dataset.targets)
weights = torch.tensor(weights).to(DEVICE)

loss_function = nn.CrossEntropyLoss(weight=weights.float())
optimizer = optim.SGD(model.fc.parameters(), lr=LR, momentum=MOMENTUM)
LR_scheduler = lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)


# TRAINING, EVALUATION, METRICS
######################################################################################

performance = []
performance_path = Path("Reports/rock_classification_performance.pkl")
tensorboard_path = Path("Reports/tensorboard_logs")
writer = SummaryWriter(log_dir=tensorboard_path) # defines the Tensorboard writer

for epoch in range(NUM_EPOCHS):
    # train
    model.train() # sets model in training mode. Turns on gradient update
    loss_train, train_labels, train_predictions = train_epoch(DEVICE,model, optimizer, loss_function,train_dataloader)
    acc_train = balanced_accuracy_score(train_labels, train_predictions)
    
    # test
    model.eval() # Freeze model weights. No model update
    loss_test, labels, predictions = test_epoch(DEVICE, model, loss_function, test_dataloader)
    acc_test = balanced_accuracy_score(labels, predictions)
    
    LR_scheduler.step()
    
    # report metrics
    current_lr = LR_scheduler.get_last_lr()
    print(f"Epoch: {epoch}. Train-loss: {loss_train:.3f}. Train-acc: {acc_train:.2f}. Test-loss: {loss_test:.3f}. Test-acc: {acc_test:.3f}. LR: {current_lr}")


    if SAVE_PERFORMANCE:
        # add data to Tensorboard for inspection of training development
        writer.add_scalars("Loss development",{
            "Loss train": loss_train,
            "Loss test": loss_test
        }, global_step=epoch)
        writer.add_scalars("Accuracy development",{
            "Accuracy train": acc_train,
            "Accuracy test": acc_test
        }, global_step=epoch)

        # save data to pickle file every epoch. Load data for later analysis of results
        performance.append({ 
        'epoch': epoch + 1,  #epoch counts from 0
        'train_loss': loss_train,
        'train_acc': acc_train,
        'test_loss': loss_test,
        'test_acc': acc_test,
        'test_labels':labels,
        'test_predictions':predictions,
        'class_names':train_dataset.classes
        })
        pickle.dump(performance, open(performance_path, 'wb'))

writer.close()
# save trained model for later predictions and analysis
if SAVE_PERFORMANCE:
    torch.save(model, Path("Reports/rock_model.pth"))