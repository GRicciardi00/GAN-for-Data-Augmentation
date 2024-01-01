#TP1 Advances in Machine Vision, Giuseppe Ricciardi 24/12/2023
#GAN-based Data Augmentation for Image Classification
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms
import os
from PIL import Image
import random
from GAN_models import DCGAN, WGAN, VAE_GAN
from torch.utils.data import DataLoader
import CNN
from CNN import CNN, loading_CNN_dataset
# Set random seed for reproducibility
manualSeed = 55
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True) # Needed for reproducible results

"""
#1. Dataset Selection
#Choose the data provided on ecampus: NewData
"""
#Spatial size of training images. All images will be resized to this size using a transformer.
image_size = 64
#Define data transformations
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
])
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, augmentations=None):
        self.root = root
        self.transform = transform
        self.augmentations = augmentations
        self.files = [file for file in os.listdir(root) if file.endswith('.png')]
        if self.augmentations:
            self.perform_augmentation()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.files[idx])
        image = Image.open(img_path)

        # Convert to RGB if the image is not in RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Infer label based on the file name
        if "cow" in self.files[idx].lower():
            label = 0
        elif "horse" in self.files[idx].lower():
            label = 1
        else:
            label = -1  # You can use any placeholder value or raise an exception

        return image, label

    def perform_augmentation(self):
        augmentation_options = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=10),
            transforms.RandomVerticalFlip(),
            transforms.RandomPerspective(distortion_scale=0.2)
        ]
        for file in self.files:
            img_path = os.path.join(self.root, file)
            original_image = Image.open(img_path).convert('RGB')
            # Randomly select one augmentation
            selected_augmentation = random.choice(augmentation_options)
            augmented_image = self.apply_transform(original_image, selected_augmentation)
            augmented_filename = f"augmented_{file}"
            augmented_root = os.path.join(self.root, "augmented")
            augmented_path = os.path.join(augmented_root, augmented_filename)
            #print(f"Saving {augmented_path}")
            augmented_image.save(augmented_path)

    def apply_transform(self, image, transform):
        if transform:
            image = transform(image)
        return image

"""
2. GAN Architecture
Implement a GAN architecture suitable for generating images similar to those in the chosen dataset.
Experiment with different GAN architectures (e.g., DCGAN, WGAN, etc.) and hyperparameters to find the most effective model.

"""
#Each different GAN architecture is defiend as a class in "GAN_models.py".

"""
3. Training GAN
Train the GAN on the original dataset to generate synthetic images. The GAN should learn to produce realistic images that resemble the distribution of the original dataset.
"""
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
#Defining hyperparameters -> Will be the same for all the GAN architectures
# Number of channels in the training images. For color images this is 3
nc = 3
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 300
# Learning rate for optimizers
lr = 0.001
# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5
#Batch size 
batch_size = 32
#Size of the validation set
val_size = 90


#MAIN
# Create datasets for both training and testing for both classes (cow and horse)
cow_train_path = "Dataset/Train/Cow/" 
cow_test_path = "Dataset/Test/Cow/"
horse_train_path = "Dataset/Train/Horse/"
horse_test_path = "Dataset/Test/Horse/"
#For the CNN, it load the dataset using ImageFolder instead of CustomDataset.
train_path = "Dataset/Train/"
test_path = "Dataset/Test/"
#User choose if he want to do data augmentation before the gan training or not.
perform_augmentation = input("Do you want to perform data augmentation? (yes/no) ").lower() == "yes"
#COW
train_cow_dataset = CustomDataset(root=cow_train_path, transform=transform)
test_cow_dataset = CustomDataset(root=cow_test_path, transform=transform)
test_cow_loader = DataLoader(test_cow_dataset, shuffle=False)
#HORS
train_horse_dataset = CustomDataset(root=horse_train_path, transform=transform)
test_horse_dataset = CustomDataset(root=horse_test_path, transform=transform)
test_horse_loader = DataLoader(test_horse_dataset, shuffle=False)

# Create the dataset with or without data augmentation based on user input
if perform_augmentation:
    #COW
    augmented_cow_dataset = CustomDataset(root=cow_train_path, transform=transform, augmentations=4)
    combined_cow_dataset = torch.utils.data.ConcatDataset([train_cow_dataset, augmented_cow_dataset])
    train_cow_loader = DataLoader(combined_cow_dataset, batch_size=32, shuffle=True)
    #HORSE
    augmented_horse_dataset = CustomDataset(root=horse_train_path, transform=transform, augmentations=4)
    combined_horse_dataset = torch.utils.data.ConcatDataset([train_horse_dataset, augmented_horse_dataset])
    train_horse_loader = DataLoader(combined_horse_dataset, batch_size=32, shuffle=True)
else:
    train_cow_loader = DataLoader(train_cow_dataset, batch_size=32, shuffle=True)
    train_horse_loader = DataLoader(train_horse_dataset, batch_size=32, shuffle=True)

#CNN Execution after data Augmentation
print("CNN execution before data augmentation...")
#train and test data directory
train_dl,val_dl,test_dl = loading_CNN_dataset(train_path, test_path,val_size, batch_size)
#Initialize the model
model = CNN(train_dl, val_dl, test_dl)
#train the model
model.do_training(num_epochs=40, lr=0.0001)
#print the training history
model.plot_training_history()
#Test the model
model.evaluate_on_test_set()
#User choose the GAN architecture to use with standard parameters
command = ""
while command != "D" and command != "W" and command != "V" and command != "0":
    command = input("Choose the GAN architecture: D for DCGAN, W for WGAN, V for VAE-GAN, 0 to skip \n")
    if command == "D":
        #DCGAN Training
        dcgan = DCGAN(nz, ngf, ndf, nc)
        #Training cow
        dcgan.train(lr, beta1, device, num_epochs, train_cow_loader, "Cow")
        dcgan.plot_train_results()
        #Training horse 
        dcgan.train(lr, beta1, device, num_epochs, train_horse_loader, "Horse")
        dcgan.plot_train_results()
        #Generate images
        dcgan.generate_image(num_images = 30,device = device, filename = "Cow")
        dcgan.generate_image(num_images = 30,device = device, filename = "Horse")

    elif command == "W":
        #WGAN Training
        wgan = WGAN(nz, ngf, ndf, nc)
        #Training cow
        wgan.train(train_cow_loader,num_epochs,lr, device, "Cow")
        wgan.plot_train_results()
        #Training horse
        wgan.train(train_horse_loader,num_epochs,lr, device, "Horse")
        wgan.plot_train_results()
        #Generate images
        wgan.generate_image(num_images = 30,device = device, filename = "Cow")
        wgan.generate_image(num_images = 30,device = device, filename = "Horse")
        
    elif command == "V":
        #VAE-GAN Training
        vae_gan = VAE_GAN(nz, ngf, ndf, nc, lr, device)
        #Training cow
        vae_gan.train(num_epochs = num_epochs, dataloader=train_cow_loader, filename ="Cow")
        #Training horse
        vae_gan.train(num_epochs = num_epochs, dataloader=train_horse_loader, filename ="Horse")       
        #Generate images
        vae_gan.generate_image(num_images = 30,device = device, filename = "Cow")
        vae_gan.generate_image(num_images = 30,device = device, filename = "Horse") 
    else:
        print("Invalid command, try again")


#CNN Execution after data Augmentation
print("CNN execution after data augmentation...")
#train and test data directory
train_dl,val_dl,test_dl = loading_CNN_dataset(train_path, test_path,val_size, batch_size)
#Initialize the model
model = CNN(train_dl, val_dl, test_dl)
#train the model
model.do_training(num_epochs=40, lr=0.0001)
#print the training history
model.plot_training_history()
#Test the model
model.evaluate_on_test_set()