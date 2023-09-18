# Import libraries
import torch
from torch import nn
import torchvision
from torchvision.io import read_image
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm.auto import tqdm
from math import ceil

# Limit non-deterministic behavior
torch.manual_seed(0)

# Define useful strings for reading images
path_str = '/home/user/Patrick/Abstract_gallery/Abstract_image_'
jpeg_str = '.jpg'

# Load all images and combine them into one useful torch tensor for training
image_channels = 3
max_RGB_value = 255
num_images = 2872
image_res = 256
image_tensor = torch.zeros(num_images, image_channels, image_res, image_res)

for i in range(num_images):
    image_num_str = str(i)
    complete_path_str = path_str + image_num_str + jpeg_str
    image = read_image(complete_path_str)
    if image.shape[1] != image.shape[2]:
        min_dim = min(image.shape[1], image.shape[2])
        image = T.CenterCrop(min_dim)(image)
    if image.shape[1] != image_res:
        image = T.Resize(image_res, antialias = True)(image)
    image = 2*(image/max_RGB_value)-1
    image_tensor[i, :, :, :] = image

# Define the generator
class Generator(nn.Module):
    '''
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        im_chan: the number of channels in the images
    '''
    def __init__(self, z_dim = 64, im_chan = 3):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        # Build the neural network
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim, 128*im_chan),
            self.make_gen_block(128*im_chan, 64*im_chan),
            self.make_gen_block(64*im_chan, 32*im_chan),
            self.make_gen_block(32*im_chan, 16*im_chan),
            self.make_gen_block(16*im_chan, 8*im_chan),
            self.make_gen_block(8*im_chan, 4*im_chan),
            self.make_gen_block(4*im_chan, 2*im_chan),
            self.make_gen_block(2*im_chan, im_chan, final_layer=True),
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size = 4, stride = 2, padding = 1, final_layer = False):
        '''
        Function to return a sequence of operations corresponding to a generator block of DCGAN,
        corresponding to a transposed convolution, a batchnorm (except for in the last layer), and an activation.
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise
                      (affects activation and batchnorm)
        '''

        # Build the neural block
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride = stride, padding = padding),
                nn.BatchNorm2d(output_channels),
                nn.ReLU()
            )
        else: # Final Layer
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride = stride, padding = padding),
                nn.Tanh()
            )

    def unsqueeze_noise(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor,
        returns a copy of that noise with width and height = 1 and channels = z_dim.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        return noise.view(len(noise), self.z_dim, 1, 1)

    def forward(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor,
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        x = self.unsqueeze_noise(noise)
        return self.gen(x)

def get_noise(n_samples, z_dim, device):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    '''
    return torch.randn(n_samples, z_dim, device = device)

# Define the critic
class Critic(nn.Module):
    '''
    Critic Class
    Values:
        im_chan: the number of channels in the images
    '''
    def __init__(self, im_chan = 3):
        super(Critic, self).__init__()
        self.crit = nn.Sequential(
            self.make_crit_block(im_chan, 2*im_chan),
            self.make_crit_block(2*im_chan, 4*im_chan),
            self.make_crit_block(4*im_chan, 8*im_chan),
            self.make_crit_block(8*im_chan, 16*im_chan),
            self.make_crit_block(16*im_chan, 32*im_chan),
            self.make_crit_block(32*im_chan, 64*im_chan),
            self.make_crit_block(64*im_chan, 128*im_chan),
            self.make_crit_block(128*im_chan, 1, final_layer=True),
        )

    def make_crit_block(self, input_channels, output_channels, kernel_size = 4, stride = 2, padding = 1, final_layer = False):
        '''
        Function to return a sequence of operations corresponding to a critic block of DCGAN,
        corresponding to a convolution, a batchnorm (except for in the last layer), and an activation.
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise
                      (affects activation and batchnorm)
        '''

        # Build the neural block
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride = stride, padding = padding),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(negative_slope=0.2)
            )
        else: # Final Layer
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride = stride, padding = padding)
            )

    def forward(self, image):
        '''
        Function for completing a forward pass of the critic: Given an image tensor,
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_dim)
        '''
        crit_pred = self.crit(image)
        return crit_pred.view(len(crit_pred), -1)

# Set up training hyperparameters
z_dim = 64
n_epochs = 10000
batch_size = 128
display_step = int(ceil(num_images/batch_size))
gen_lr = 0.002
crit_lr = 0.002
beta_1 = 0.5
beta_2 = 0.999
c_lambda = 10
crit_repeats = 5
device = "cuda"

# Initialize generator and critic
gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=gen_lr, betas=(beta_1, beta_2))
crit = Critic().to(device)
crit_opt = torch.optim.Adam(crit.parameters(), lr=crit_lr, betas=(beta_1, beta_2))

# Initialize the weights to the normal distribution
# with mean 0 and standard deviation 0.02
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
gen = gen.apply(weights_init)
crit = crit.apply(weights_init)

# Define a function to calculate the gradient that we want to obey 1-Lipschitz continuity
def get_gradient(crit, real_batch, fake_batch, epsilon):
    '''
    Return the gradient of the critic's scores with respect to mixes of real and fake images.
    Parameters:
        crit: the critic model
        real_batch: a batch of real images
        fake_batch: a batch of fake images
        epsilon: a vector of the uniformly random proportions of real/fake per mixed image
    Returns:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    '''
    # Mix the images together
    mixed_images = real_batch * epsilon + fake_batch*(1-epsilon)

    # Calculate the critic's scores on the mixed images
    mixed_scores = crit(mixed_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient

# Define the loss function that penalizes violation of 1-Lipschitz continuity
def gradient_penalty(gradient):
    '''
    Return the gradient penalty, given a gradient.
    Given a batch of image gradients, you calculate the magnitude of each image's gradient
    and penalize the mean quadratic distance of each magnitude to 1.
    Parameters:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    Returns:
        penalty: the gradient penalty
    '''
    # Flatten the gradients so that each row captures one image
    gradient = gradient.view(len(gradient), -1)

    # Calculate the magnitude of every row
    gradient_norm = gradient.norm(2, dim=1)

    # Penalize the mean squared distance of the gradient norms from 1
    penalty = torch.mean((gradient_norm-1)**2)
    return penalty

# Define the total loss function for the generator
def get_gen_loss(crit_fake_pred):
    '''
    Return the loss of a generator given the critic's scores of the generator's fake images.
    Parameters:
        crit_fake_pred: the critic's scores of the fake images
    Returns:
        gen_loss: a scalar loss value for the current batch of the generator
    '''
    gen_loss = -torch.mean(crit_fake_pred)
    return gen_loss

# Define the total loss function for the critic
def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):
    '''
    Return the loss of a critic given the critic's scores for fake and real images,
    the gradient penalty, and gradient penalty weight.
    Parameters:
        crit_fake_pred: the critic's scores of the fake images
        crit_real_pred: the critic's scores of the real images
        gp: the unweighted gradient penalty
        c_lambda: the current weight of the gradient penalty
    Returns:
        crit_loss: a scalar for the critic's loss, accounting for the relevant factors
    '''
    crit_loss = torch.mean(crit_fake_pred) - torch.mean(crit_real_pred) + c_lambda*gp
    return crit_loss

# Train
cur_step = 0
generator_losses = []
critic_losses = []
batches = DataLoader(image_tensor, batch_size = batch_size, shuffle = True)
for epoch in range(n_epochs):
    for real_batch in tqdm(batches):
        cur_batch_size = len(real_batch)
        real_batch = real_batch.to(device)

        mean_iteration_critic_loss = 0
        for _ in range(crit_repeats):
            ### Update critic ###
            crit_opt.zero_grad()
            fake_noise = get_noise(cur_batch_size, z_dim, device = device)
            fake_batch = gen(fake_noise)
            crit_fake_pred = crit(fake_batch.detach())
            crit_real_pred = crit(real_batch)

            epsilon = torch.rand(len(real_batch), 1, 1, 1, device = device, requires_grad = True)
            gradient = get_gradient(crit, real_batch, fake_batch.detach(), epsilon)
            gp = gradient_penalty(gradient)
            crit_loss = get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda)

            # Keep track of the average critic loss in this batch
            mean_iteration_critic_loss += crit_loss.item() / crit_repeats
            # Update gradients
            crit_loss.backward(retain_graph = True)
            # Update optimizer
            crit_opt.step()
        critic_losses += [mean_iteration_critic_loss]

        ### Update generator ###
        gen_opt.zero_grad()
        fake_noise_2 = get_noise(cur_batch_size, z_dim, device)
        fake_2 = gen(fake_noise_2)
        crit_fake_pred = crit(fake_2)

        gen_loss = get_gen_loss(crit_fake_pred)
        gen_loss.backward()

        # Update the weights
        gen_opt.step()

        # Keep track of the average generator loss
        generator_losses += [gen_loss.item()]

        ### Visualization code ###
        if cur_step % display_step == 0 and cur_step > 0:
            gen_mean = sum(generator_losses[-display_step:]) / display_step
            crit_mean = sum(critic_losses[-display_step:]) / display_step
            print("Epoch:", epoch, "Step:", cur_step, "Generator Loss:", gen_mean, "Critic Loss:", crit_mean)

        cur_step += 1
gen.eval()
crit.eval()
torch.save(gen.state_dict(), '/home/user/Patrick/model_WGAN_gen.pt')
torch.save(crit.state_dict(), '/home/user/Patrick/model_WGAN_crit.pt')
