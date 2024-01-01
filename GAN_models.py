import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torchvision.utils import save_image
from torchvision.utils import make_grid
import torch.nn.functional as F
import torch.nn.init as init
#First class: DCGAN, Deep Convolutional Generative Adversarial Networks. It is a GAN architecture made of convolutional layers.
#The generator is made of convolutional transpose layers, while the discriminator is made of convolutional layers.
class DCGAN (nn.Module):
    def __init__(self, nz, ngf, ndf, nc):
        super(DCGAN, self).__init__()
        #nz is the size of the latent z vector
        self.nz = nz
        #ngf is the size of feature maps in the generator
        self.ngf = ngf
        #ndf is the size of feature maps in the discriminator
        self.ndf = ndf
        #nc is the number of channels in the training images. For color images this is 3
        self.nc = nc
        #Generator
        self.generator = nn.Sequential(
            # input is Z, going into a convolution
            # state size. ngf*8 is the number of channels in the output image, 4 is the kernel size and 4 is the stride size (4x4).
            nn.ConvTranspose2d( self.nz, self.ngf * 8, 4, 1, 0, bias=False),
            #Batch normalization is a technique for improving the speed, performance, and stability of artificial neural networks. It is used to normalize the input layer by adjusting and scaling the activations. 
            #This is done by calculating the mean and variance of the input layer, then applying the normalization equation to each input.
            nn.BatchNorm2d(self.ngf * 8),
            #Use the rectified linear unit function as the activation function. The rectified linear activation function is a piecewise linear function that will output the input directly if is positive, otherwise, it will output zero.
            nn.ReLU(True),
            #Repeat the same process as before, but with different number of channels in the output image. The number of channels in the output image is halved at each step.
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d( self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d( self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d( self.ngf, self.nc, 4, 2, 1, bias=False),
            #Last activation function is a Tanh function, a type of activation function that is similar to the sigmoid. It is used to normalize the output of a network to a range between -1 and 1.
            nn.Tanh()
        )
        #Discriminator
        self.discriminator = nn.Sequential(
            #Input is ``(nc) x 64 x 64``, where nc is the number of channels in the input image, 64 is the height and 64 is the width.
            #Convolutional layer with 3 input channels, 64 output channels, 4 kernel size and 2 stride size (4x4). 
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            #Use the Leaky ReLU function as the activation function. The Leaky ReLU function is an improved version of the ReLU function. The Leaky ReLU function addresses the “dying ReLU” problem
            #by having a small negative slope in the negative section, instead of altogether zero.
            nn.LeakyReLU(0.2),
            #Repeat the same process as before, but with different number of channels in the output image. The number of channels in the output image is doubled at each step.
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            #Batch normalization applied to the output of the convolutional layer as for the generator.
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            #Last activation function is a Sigmoid function to maps input values to probabilities between 0 and 1.
            nn.Sigmoid()
        )
        self.G_losses = []
        self.D_losses = []
        self.img_list = []
        
    #forward_generator and forward_discriminator are used to call the generator and the discriminator.
    def forward_generator(self, input):
        return self.generator(input)
    def forward_discriminator(self, input):
        return self.discriminator(input)
    #custom weights initialization called on generator and discriminator 
    def weights_init(self, m):
        classname = m.__class__.__name__
        #If the module is a Convolution, initialize the weight and bias tensors with normal distribution centered at 0.0 with a standard deviation of 0.02.
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        #If the module is a BatchNorm2d, initialize the weight tensor as a normal distribution with mean 1.0 and standard deviation 0.02, and bias tensor as 0. 
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    #train function is used to train the GAN model
    def train(self, lr, beta1, device, num_epochs, train_loader, filename):
        # Initialize BCELoss function
        #Binary Cross Entropy Loss function
        criterion = nn.BCELoss()
        #Initialize optimizers, one for generator and one for discriminator. The optimizer will be Adam, algorithm for first-order gradient-based optimization of stochastic objective functions.
        optimizerDCGAN_D = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.8))
        optimizerDCGAN_G = optim.Adam(self.generator.parameters(), lr=lr-0.00004, betas=(beta1-0.06, 0.6))
        # Create batch of latent vectors that we will use to visualize the progression of the generator
        fixed_noise = torch.randn(64, self.nz, 1, 1, device=device)
        # Establish convention for real and fake labels during training
        real_label = 1
        fake_label = 0
        ## Training Loop
        #initialize weight
        self.generator.apply(self.weights_init)
        self.discriminator.apply(self.weights_init)
        print("Starting DGAN Training Loop...")
        # For each epoch
        for epoch in range(num_epochs):
            # For each batch in the dataloader
            for i, data in enumerate(train_loader, 0):
                #Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                # Train with all-real batch: maximize log(D(x)), 
                # Calculate the gradients for this batch
                self.discriminator.zero_grad()
                # Format batch -> data[0] is the image, data[1] is the label
                # data[0] is a tensor of size (batch_size, 3, 64, 64), where 3 is the number of channels in the image, 64 is the height and 64 is the width
                real_cpu = data[0].to(device)
                # b_size is the batch size of the real_cpu tensor 
                b_size = real_cpu.size(0)
                # Accumulate real images
                label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
                # Forward pass real batch through D
                # output is a tensor of size (batch_size, 1, 1, 1)
                output = self.discriminator(real_cpu).view(-1)
                # Calculate loss on all-real batch
                errD_real = criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()
                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, self.nz, 1, 1, device=device)
                # Generate fake image batch with G
                fake = self.generator(noise)
                label.fill_(fake_label)
                # Classify all fake batch with D
                output = self.discriminator(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = criterion(output, label)
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_fake
                # Update Discriminator, step() method updates the parameters of the model using the gradients computed by the backward() method. 
                optimizerDCGAN_D.step()
                # Update G network: maximize log(D(G(z)))
                self.generator.zero_grad()
                label.fill_(real_label) # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = self.discriminator(fake).view(-1)
                # Calculate G's loss based on this output
                errG = criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                optimizerDCGAN_G.step()
                # Save real images periodically or at the end of each epoch
                # Output training stats
                if i % 2 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                        % (epoch, num_epochs, i, len(train_loader),
                            errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                # Save Losses for plotting later
                self.G_losses.append(errG.item())
                self.D_losses.append(errD.item())
                #Save training images of first epoch, middle epoch, last epoch 

                if (epoch == 0 and i == 0) or (epoch == num_epochs-1 and i == len(train_loader)-1) or (epoch == num_epochs/2 and i == len(train_loader)/2):
                    with torch.no_grad():
                        fake_images = self.forward_generator(fixed_noise).detach().cpu()
                    save_image(fake_images, f"IMG_DCGAN/{filename}-epoch_{epoch}_batch_{i}_fake.png", normalize=True)
                    save_image(real_cpu, f"IMG_DCGAN/{filename}-epoch_{epoch}_batch_{i}_real.png", normalize=True)
        #Save the model
        torch.save(self, f'pth/{filename}-dcgan.pth')

    def plot_train_results(self):
        plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.G_losses,label="Generator")
        plt.plot(self.D_losses,label="Discriminator")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def generate_image(self, num_images,device,filename):
        #Load model
        model = torch.load(f'pth/{filename}-dcgan.pth')
        #Generate num_images images
        for i in range(num_images):
            with torch.no_grad():
                noise = torch.randn(1, 100, 1, 1, device=device)
                fake = model.forward_generator(noise).detach().cpu()
            #Save images
            save_image(fake, f"./Dataset/Train/{filename}/{filename}-generated-dcgan-{i+1}.png", normalize=True)

def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0)

#Second class: WGAN, Wasserstein Generative Adversarial Networks. It is a GAN architecture made of convolutional layers, like the DCGAN. 
#The architecture is the same as the DCGAN, but the loss function is different. The loss function is the Wasserstein loss function
class WGAN(nn.Module):
    def __init__(self, nz, ngf, ndf, nc):
        super(WGAN, self).__init__()
        #ngpu is the number of GPUs available. If this is 0, code will run in CPU mode.
        #nz is the size of the latent z vector
        self.nz = nz
        #ngf is the size of feature maps in the generator
        self.ngf = ngf
        #ndf is the size of feature maps in the discriminator
        self.ndf = ndf
        #nc is the number of channels in the training images. For color images this is 3
        self.nc = nc
        #Generator
        self.generator = nn.Sequential(
            # input is Z, going into a convolution
            # state size. ``(ngf*8) x 4 x 4``, where 4 is the kernel size and 4 is the stride size (4x4) and ngf*8 is the number of channels in the output image 
            nn.ConvTranspose2d( self.nz, self.ngf * 8, 4, 1, 0, bias=False),
            #Batch normalization is a technique for improving the speed, performance, and stability of artificial neural networks. It is used to normalize the input layer by adjusting and scaling the activations. 
            #This is done by calculating the mean and variance of the input layer, then applying the normalization equation to each input.
            nn.BatchNorm2d(self.ngf * 8),
            #Use the rectified linear unit function as the activation function. The rectified linear activation function is a piecewise linear function that will output the input directly if is positive, otherwise, it will output zero.
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d( self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d( self.ngf, self.nc, 4, 2, 1, bias=False),
            #Last activation function is a Tanh function. The Tanh function is a hyperbolic tangent function, a type of activation function that is similar to the sigmoid.
            nn.Tanh()
        )
        #Discriminator
        self.discriminator = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            #Use the Leaky ReLU function as the activation function. The Leaky ReLU function is an improved version of the ReLU function. The Leaky ReLU function addresses the “dying ReLU” problem
            #by having a small negative slope in the negative section, instead of altogether zero. Usually, the slope is set to 0.2 
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.G_losses = []
        self.D_losses = []
        self.img_list = []

    #forward_generator and forward_discriminator are used to call the generator and the discriminator.
    def forward_generator(self, input):
        return self.generator(input)
    def forward_discriminator(self, input):
        return self.discriminator(input)
    def train(self, dataloader, num_epochs,lr, device, filename):
        fixed_noise = torch.randn(64, self.nz, 1, 1, device=device)
        #optimizers for generator and discriminator, RMSprop is an optimizer that utilizes the magnitude of recent gradients to normalize the gradients.
        optimizer_G = optim.RMSprop(self.generator.parameters(), lr=lr-0.00004)
        optimizer_D = optim.RMSprop(self.discriminator.parameters(), lr=lr)
        #Initialize model weights
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
        for epoch in range(num_epochs):
            for i, data in enumerate(dataloader, 0):
                real_cpu = data[0].to(device)
                b_size = real_cpu.size(0)

                # Train Discriminator
                optimizer_D.zero_grad()

                # Forward pass real batch through D
                output_real = self.discriminator(real_cpu).view(-1)
                # Calculate loss on all-real batch (Wasserstein discriminator loss) -> -E[D(x)] + E[D(G(z))] 
                errD_real = -torch.mean(output_real)

                # Generate fake batch with G
                noise = torch.randn(b_size, 100, 1, 1, device= device)
                fake = self.generator(noise)

                # Forward pass fake batch through D
                output_fake = self.discriminator(fake.detach()).view(-1)
                errD_fake = torch.mean(output_fake)

                # Wasserstein discriminator loss
                errD = errD_real + errD_fake
                errD.backward()
                optimizer_D.step()

                # Clip weights of discriminator (WGAN weight clipping)
                # The weight clipping is a parameter of the WGAN algorithm. It is used to enforce the Lipschitz constraint on the discriminator.
                # The Lipschitz constraint is a condition that ensures the existence of a gradient for the discriminator loss function.
                # The weight clipping is implemented by clipping the weights of the discriminator to a small fixed range after each update.
                for p in self.discriminator.parameters():
                    p.data.clamp_(-0.01, 0.01)

                # Train Generator
                optimizer_G.zero_grad()

                # Forward pass fake batch through D
                output = self.discriminator(fake).view(-1)
                errG = -torch.mean(output)
                errG.backward()
                optimizer_G.step()

                # Print statistics
                if i % 2 == 0:
                    print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
                          % (epoch, num_epochs, i, len(dataloader), errD.item(), errG.item()))
                if (epoch == 0 and i == 0) or (epoch == num_epochs-1 and i == len(dataloader)-1) or (epoch == num_epochs/2 and i == len(dataloader)/2):
                    with torch.no_grad():
                        fake_images = self.forward_generator(fixed_noise).detach().cpu()
                    save_image(fake_images, f"IMG_WGAN/{filename}-epoch_{epoch}_batch_{i}_fake.png", normalize=True)
                    save_image(real_cpu, f"IMG_WGAN/{filename}-epoch_{epoch}_batch_{i}_real.png", normalize=True)
                # Save Losses for plotting later
                self.G_losses.append(errG.item())
                self.D_losses.append(errD.item())
        #Save the model
                torch.save(self, f'pth/{filename}-wgan.pth')

    def plot_train_results(self):
        #Plot loss
        plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.G_losses,label="Generator")
        plt.plot(self.D_losses,label="Discriminator")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
    
    def generate_image(self, num_images,device,filename):
        #Load model
        model = torch.load(f'pth/{filename}-wgan.pth')
        #Generate num_images images
        for i in range(num_images):
            with torch.no_grad():
                noise = torch.randn(1, 100, 1, 1, device=device)
                fake = model.forward_generator(noise).detach().cpu()
            #Save images
            save_image(fake, f"./Dataset/Train/{filename}/{filename}-generated-wgan-{i+1}.png", normalize=True)
        
            

#Third class: VAE-GAN, Variational Autoencoder Generative Adversarial Networks.
#It's an architecture that combines the VAE and the GAN architectures.
class VAE(nn.Module):
    def __init__(self, nz, nc, device):
        super(VAE, self).__init__()
        self.device = device
        self.nz = nz
        self.nc = nc
        self.encoder = nn.Sequential(
            nn.Conv2d(self.nc, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d((2, 2)),
        )
        self.encoder_fc1 = nn.Linear(64 * 8 * 8, self.nz)
        self.encoder_fc2 = nn.Linear(64 * 8 * 8, self.nz)
        self.decoder_fc = nn.Linear(self.nz, 64 * 8 * 8)  # Adjusted output size
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=3, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, self.nc, kernel_size=3, stride=3, padding=1, output_padding=0),
            nn.Sigmoid(),
        )


    def forward(self, x):
        out = self.encoder(x)
        out = out.view(out.size(0), -1)
        mean = self.encoder_fc1(out)
        logstd = self.encoder_fc2(out)
        z = self.noise_reparameterize(mean, logstd)
        out3 = self.decoder_fc(z)
        out3 = out3.view(out3.size(0), 64, 8, 8)  # Adjusted dimensions
        # Avoid inplace operation by using the result directly in the next line
        return self.decoder(out3), mean, logstd

    def noise_reparameterize(self, mean, logvar):
        eps = torch.randn(mean.shape).to(self.device)
        z = mean + eps * torch.exp(logvar)
        return z



class VAE_GAN (nn.Module):
    def __init__(self, nz, ngf, ndf, nc, lr,device,):
        super(VAE_GAN, self).__init__()
        #nz is the size of the latent z vector
        self.nz = nz
        #ngf is the size of feature maps in the generator
        self.ngf = ngf
        #ndf is the size of feature maps in the discriminator
        self.ndf = ndf
        #nc is the number of channels in the training images. For color images this is 3
        self.nc = nc
        self.device = device
        #VAE part
        self.vae = VAE(self.nz,self.nc,self.device)
        #GAN part (Discriminator)
        #Discriminator is composed of convolutional layers. 
        #The discriminator takes as input the reconstructed images from the VAE and the real images and classifies them as real or fake.
        self.dis = nn.Sequential(
            nn.Conv2d(self.nc, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((2, 2)),
        )
        self.fc = nn.Sequential(
            nn.Linear(8 * 8 * 64, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )       
        self.optimizerV = optim.Adam(self.vae.parameters(), lr=lr-0.00006)
        self.optimizerD = optim.Adam(self.dis.parameters(), lr=lr)
        self.Dlosses = []
        self.Vaelosses = []
    def forward(self, input):
        x = self.dis(input)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.squeeze(1)

    def loss_function(self, recon_x, x, mean, logstd):
        # Use non-inplace operations to avoid gradient computation issues
        # Compute Mean Squared Error (MSE) loss
        MSECriterion = nn.MSELoss().to(self.device)
        recon_x_resized = F.interpolate(recon_x, size=x.size()[2:], mode='bilinear', align_corners=False)
        MSE = MSECriterion(recon_x_resized, x)
        # Because var is a log-variance, we need to exponentiate it to get the variance
        var = torch.pow(torch.exp(logstd), 2)
        # Compute Kullback-Leibler Divergence (KLD) loss
        KLD = -0.5 * torch.sum(1 + torch.log(var) - torch.pow(mean, 2) - var)
        return MSE + KLD
    def train(self, num_epochs, dataloader, filename):
        criterion = nn.BCELoss().to(self.device)
        print("Begin training of VAE-GAN")
        #Wheight initialization:
        self.vae.apply(weights_init)
        self.dis.apply(weights_init)
        for epoch in range(num_epochs):
            for i, (data, _) in enumerate(dataloader, 0):
                # Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                # train with real
                self.dis.zero_grad()
                data = data.to(self.device)
                batch_size = data.shape[0]
                output = self.dis(data)

                # Create target tensor for real images
                real_label = torch.full((batch_size,), 1.0, device=self.device)
                real_label = real_label.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(output)
                errD_real = criterion(torch.sigmoid(output), real_label)  # Apply sigmoid to the output
                errD_real.backward()
                real_data_score = output.mean().item()

                # train with fake
                z = torch.randn(batch_size, self.nz).to(self.device)
                fake_data = self.vae.decoder_fc(z).view(z.shape[0], 64, 8, 8)  # Adjusted dimensions
                fake_data = self.vae.decoder(fake_data)
                output = self.dis(fake_data.detach())

                # Create target tensor for fake images
                fake_label = torch.full((batch_size,), 0.0, device=self.device)
                fake_label = fake_label.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(output)
                errD_fake = criterion(torch.sigmoid(output), fake_label)  # Apply sigmoid to the output
                errD_fake.backward()
                fake_data_score = output.mean().item()

                errD = errD_real + errD_fake
                torch.nn.utils.clip_grad_norm_(self.dis.parameters(), max_norm=1.0)
                self.optimizerD.step()

                # Update G network which is the decoder of VAE
                recon_data, mean, logstd = self.vae(data)
                self.vae.zero_grad()
                vae_loss = self.loss_function(recon_data, data, mean, logstd)
                vae_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=1.0)
                self.optimizerV.step()

                # Update G network: maximize log(D(G(z)))
                self.dis.zero_grad()
                real_label = torch.full((batch_size,), 1.0, device=self.device)
                output = self.dis(recon_data.detach())  # Detach to avoid inplace operation

                # Create target tensor for real images (reconstructed)
                real_label = real_label.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(output)
                errVAE = criterion(torch.sigmoid(output), real_label)  # Apply sigmoid to the output
                errVAE.backward()
                self.optimizerV.step()
                # Output training stats
                self.Dlosses.append(errD.item())
                self.Vaelosses.append(vae_loss.item())
                if i % 5 == 0:
                    print('[%d/%d][%d/%d] real_score: %.4f fake_score: %.4f '
                        'Loss_D: %.4f Loss_VAE: %.4f' 
                        % (epoch, num_epochs, i, len(dataloader),
                            real_data_score,
                            fake_data_score,
                            errD.item(), 
                            errVAE.item()))
                if (epoch == 0 and i == 0) or (epoch == num_epochs-1 and i == len(dataloader)-1) or (epoch == num_epochs/2 and i == len(dataloader)/2):
                    sample = torch.randn(80, self.nz).to(self.device)
                    output = self.vae.decoder_fc(sample)
                    output = self.vae.decoder(output.view(output.shape[0], 64, 8, 8))
                    fake_images = make_grid(output.cpu(), nrow=8, normalize=True).detach()
                    save_image(fake_images, f'./img_VAE-GAN/{filename}-fake_images-{epoch + 1}.png')

        torch.save(self.vae.state_dict(), f'./pth/VAE-GAN-VAE-{filename}.pth')
        torch.save(self.dis.state_dict(), f'./pth/VAE-GAN-Discriminator-{filename}.pth')
        torch.save(self, f'./pth/VAE-GAN-{filename}.pth')
        # Plot loss
        plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.Vaelosses,label="Generator")
        plt.plot(self.Dlosses,label="Discriminator")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
    def generate_image(self, num_images,device,filename):
        #Load model
        model = torch.load(f'pth/VAE-GAN-{filename}.pth')
        #Generate num_images images
        for i in range(num_images):
            with torch.no_grad():
                noise = torch.randn(1, 100, 1, 1, device=device)
                fake = model.forward_generator(noise).detach().cpu()
                #Save images
                save_image(fake, f"./Dataset/Train/{filename}/{filename}-generated-vae_gan-{i+1}.png", normalize=True)
        

    

