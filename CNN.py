import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torchvision.utils import make_grid
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn.init as init
from torchvision.datasets import ImageFolder
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


def to_np(img_tensor):
    img_np = img_tensor.numpy().transpose((1, 2, 0))
    return img_np

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
  
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

  
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

  
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

  
def fit(epochs, lr, model, train_loader, val_loader, opt_func = torch.optim.SGD):
    
    history = []
    optimizer = opt_func(model.parameters(),lr)
    for epoch in range(epochs):
        
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    
    return history


class ImageClassificationBase(nn.Module):
    
    def training_step(self, batch):
            images, labels = batch 
            out = self(images)                  # Generate predictions
            loss = F.cross_entropy(out, labels) # Calculate loss
            return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))

#Class for CNN model used in main.py for classification
#Input: 64x64x3 images
#Output: 1x1x1 probability of being cow or horse
class CNN(ImageClassificationBase):
    def __init__(self, train_dl, val_dl, test_dl):
        super().__init__()
        self.network = nn.Sequential(
            
            nn.Conv2d(3, 32, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(32,64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(128 ,128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(256,256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Flatten(),
            nn.Linear(256*8*8,1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512,2)
        )
        #Initialize weights using Kaiming initialization
        self.apply(self._init_weights)
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl
        #Weight initialization
        self.apply(self._init_weights)
    def forward(self, xb):
        return self.network(xb)

    def _init_weights(self, layer):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            if layer.bias is not None:
                init.constant_(layer.bias, 0)

    def do_training(self, num_epochs, lr):
        optimizer = torch.optim.Adam
        self.history = fit(num_epochs, lr, self, self.train_dl, self.val_dl, optimizer)

    def evaluate_on_test_set(self):
        self.eval()
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for images, labels in self.test_dl:
                outputs = self(images)
                _, preds = torch.max(outputs, 1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(preds.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_predictions)
        classification_rep = classification_report(all_labels, all_predictions)
        confusion_mat = confusion_matrix(all_labels, all_predictions)

        print("Test Set Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:\n", classification_rep)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Cow", "Horse"], yticklabels=["Cow", "Horse"])
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        plt.show()
    
    def plot_training_history(self):
        self.plot_accuracies()
        self.plot_losses()
        
    def plot_accuracies(self):
        accuracies = [x['val_acc'] for x in self.history]
        plt.plot(accuracies)
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('Accuracy vs. No. of epochs')
        plt.show()

    def plot_losses(self):
        train_losses = [x.get('train_loss') for x in self.history]
        val_losses = [x['val_loss'] for x in self.history]
        plt.plot(train_losses)
        plt.plot(val_losses)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['Training', 'Validation'])
        plt.title('Loss vs. No. of epochs')
        plt.show()

def show_batch(dl):
    """Plot images grid of single batch"""
    for images, labels in dl:
        fig,ax = plt.subplots(figsize = (16,12))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images,nrow=16).permute(1,2,0))
        plt.show()
        break

def loading_CNN_dataset(train_path, test_path,val_size, batch_size):
    #load the train and test data
    dataset = ImageFolder(train_path,transform = transforms.Compose([
        transforms.Resize((64,64)),transforms.ToTensor()
    ]))
    test_dataset = ImageFolder(test_path,transforms.Compose([
        transforms.Resize((64,64)),transforms.ToTensor()
    ]))
    img, label = dataset[0]
    print(img.shape,label)
    print("Follwing classes are there: \n", dataset.classes)
    print("Dataset length: ", len(dataset))
        # Create an array of labels
    labels = [label for _, label in dataset]
    # Use stratified sampling to split the data
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=42)

    # Get the indices for training and validation sets
    train_indices, val_indices = next(stratified_split.split(labels, labels))

    # Create DataLoader using SubsetRandomSampler to sample based on indices
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # Create DataLoaders
    train_dl = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_dl = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    test_dl = DataLoader(test_dataset,batch_size, pin_memory = True)

    #Count the number of images in each class in the train and validation set
    cow_count = 0
    horse_count = 0
    #Training test
    for images, labels in train_dl:
        for label in labels:
            if label == 0:
                cow_count += 1
            else:
                horse_count += 1
    print("Horse in train set: ", horse_count)
    print("Cow in train set: ", cow_count)
    #Validation set
    cow_count = 0
    horse_count = 0
    for images, labels in val_dl:
        for label in labels:
            if label == 0:
                cow_count += 1
            else:
                horse_count += 1
    print("Horse in validation set: ", horse_count)
    print("Cow in validation set: ", cow_count)
    #Test set
    cow_count = 0
    horse_count = 0
    for images, labels in test_dl:
        for label in labels:
            if label == 0:
                cow_count += 1
            else:
                horse_count += 1
    print("Horse in test set: ", horse_count)
    print("Cow in test set: ", cow_count)
    return train_dl, val_dl, test_dl

