import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 100 # Set this to 300 to get better image quality
from PIL import Image # We use PIL to load images
import seaborn as sns
import imageio # to generate .gifs
from sklearn import preprocessing as prs

# always good to have
import glob, random, os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
# the typical pytorch imports
import torch
import torchvision
from torchvision import transforms 
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
from torch.nn import Conv2d
from torch.optim import Adam
from torch.nn import init

from torchvision.utils import save_image

"""### Hyperparameters

These are just random hyperparameters, play around with them.
"""

IMG_SIZE = 64   # We assume images are square with IMG_SIZE*IMG_SIZE pixels, 16x16 works, too.
EPOCHS = 4000
BATCH_SIZE = 64
BETA_START = 0.0001  # The Variance scheduler constant
BETA_END = 0.05    # The Variance scheduler constant
experiment_name = 'new_pred_8_a'
# On Colab, go to Runtime -> Change runtime type to switch to GPU 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHANNELS = 3  # We use RGB images
# Experiments Type
TIMESTEPS = 300  # we will try for time steps 100 / 300
TIME_DIV = int(TIMESTEPS//10) # This corresponds to the time steps after which we save the image
LOSS_TYPE = "l2" 
NOISE_SCHEDULER_TYPE = "linear" # we will try for linear/ quadratic/ sigmoid
TIME_EMBEDDING_TYPE = "linear" # we will try for sinusioidal /  linear
NUM_GROUPS = 8   # for group normalization groups. Check for 8 / 32
LEARNING_RATE=1e-4
NUM_RES_BLOCKS = 2 
DROPOUT = 0.1    
TIME_DIMENSION= 1 # for time_embed type: linear =1 , sinusoidal=128 
CLASS_COND = None  # For using pokemon type conditioning use 'yes' or else use None   
FEAT_COND = None   # For using image feature conditioning use 'yes' or else use None 
DEVICE

# Create the various faolders to save the results

# save intermediate results during training
results_folder = Path("./intermediate_results/{}".format(experiment_name))
results_folder.mkdir(exist_ok = True)

# save the trained weights based on the experiment. 
# Basically we will be saving two weights: best and the last
outdir = Path("./exps/{}".format(experiment_name))
outdir.mkdir(exist_ok=True)

#  Save the generated results during inference. Also save the backward diffusion gif file here
save_img_dir = Path("./saved_imgs/{}".format(experiment_name))
save_img_dir.mkdir(exist_ok=True)

# #import zipfile module
# from zipfile import ZipFile

# with ZipFile('./pokemon_temp/pokemon.zip', 'r') as f:

#     #extract in current directory
#     f.extractall()

np.random.seed(42)
torch.manual_seed(42)
torch.random.manual_seed(42)
random.seed(42)
torch.cuda.manual_seed_all(42)
os.environ['PYTHONHASHSEED'] = '42'
torch.backends.cudnn.deterministic = True

from torchmetrics.image.fid import FrechetInceptionDistance as FID

"""### Utils

We start by defining some friendly little helpers.
"""

def image_to_tensor(image):
    """Convert a PIL image to a PyTorch tensor.

    Args:
        image (PIL.Image): The image to be converted.

    Returns:
        torch.Tensor: The converted PyTorch tensor.
    """
    
    # Define the image transformation pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Lambda(lambda t: (t - 0.5) * 2.)
    ])

    # Apply the transformation pipeline to the image
    return transform(image)


def tensor_to_img(tensor):
    """Convert a PyTorch tensor to a PIL image.

    Args:
        tensor (torch.Tensor): The PyTorch tensor to be converted.

    Returns:
        PIL.Image: The converted PIL image.
    """
    # print(tensor)
    # tensor,lbl = tensor_tuple
    # Define the tensor transformation pipeline
    transform = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1.) / 2.),
        transforms.Lambda(lambda t: torch.clamp(t, min=0., max=1.)),
        transforms.ToPILImage()
    ])

    # Apply the transformation pipeline to the tensor
    return transform(tensor)#,lbl


def show_from_image_list(img_list, img_num=10, filename=None):
    """Show a grid of images from a list of images. Sample uniformly spaced images.
        (Alternativly, use make_grid from torchvision.utils)

    Args:
        img_list (list): The list of images to be displayed. Can be PIL or image tensors.
        img_num (int, optional): The number of images to be displayed. Default is 10.
        filename (str, optional): The name of the file to save the plot to. If None, the plot will not be saved. Default is None.
    """
    # Ensure that the number of images to be displayed is less than or equal to the number of images in the list
    img_num = min(len(img_list), img_num)

    # Get the index of the images to be displayed
    img_num_index = np.linspace(0, len(img_list)-1.0, num=img_num).astype(int)

    # Clear the current figure (if there is any)
    plt.clf()

    # Create a figure with 1 row and `img_num` columns
    fig, ax = plt.subplots(1, img_num, figsize=(15, 15), gridspec_kw={'width_ratios': [1] * img_num})

    # Iterate through the images to be displayed
    for i, idx in enumerate(img_num_index):
        img_i = img_list[idx]

        # Check if the image is a PyTorch tensor and convert it to PIL
        if isinstance(img_i, torch.Tensor):
            img_i = tensor_to_img(img_i)

        # Display the image
        ax[i].imshow(img_i)
        ax[i].axis('off')

    # Save the plot to a file if the filename is provided
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()

"""# Defining the Dataset and Loader

First, we need to create a `Dataset` and a `DataLoader` containing the images (currently in the image folder). This is PyTorch best practice.

The `Dataset` provides access to individual data samples/images  (also providing on-the-fly image augmentation). The `Dataloader` enables efficient loading of the data in mini-batches (also providing randomization).

We build our own `Datset` class that loads the images from disk and stores them in PyTorch tensors of shape $(3, \text{IMG_SIZE}, \text{IMG_SIZE})$ with values between $-1$ and $1$.
"""

from torch.utils.data import Dataset, DataLoader, IterableDataset, TensorDataset


class ImageDataset(Dataset):
    """A dataset for images that supports on-the-fly transformations (i.e., augmentations).

      Args:
          imgpaths (str, optional): The path pattern for the images. Default is "pokemon/*png".
      """

    def __init__(self, img_paths="./pokemon/*png",label_path ='./pokemon.csv'):

        # You can/should play around with these. Which of the augmantation make sense? 
        self.on_the_fly_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
#         transforms.RandomApply(torch.nn.ModuleList([transforms.RandomCrop(int(IMG_SIZE*0.8)),]), p=0.1),
#         transforms.RandomAutocontrast(p=0.1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Lambda(lambda t: torch.clamp(t, min=-1., max=1.)),
        ])

        self.img_list = list()
        # Read the labels from csv        
        # Complete label list. First get string labels converted to integer values
        df1 = pd.read_csv(label_path, delimiter=',')
        # df1.dataframeName = 'pokemon.csv'
        nRow, nCol = df1.shape
        print(f'There are {nRow} rows and {nCol} columns')
        self.label_string_list = list(df1['type'])
        le = prs.LabelEncoder()
        le.fit(self.label_string_list)
        self.classes = list(le.classes_)
        print(f'There are {len(self.classes)} classes')
        self.label_list = list(le.transform(self.label_string_list))

        for img_path in glob.glob(img_paths):
          # Turn the transparent part in the image to white following 
          # https://stackoverflow.com/questions/50898034/how-replace-transparent-with-a-color-in-pillow
            image = Image.open(img_path)
            rgba_image = Image.new("RGBA", image.size, "WHITE")
            rgba_image.paste(image, (0, 0), image)   
            rgb_image = rgba_image.convert('RGB')

              # Convert the PIL image to a tensor, where each value is in [-1,1].
            img_as_tensor = image_to_tensor(rgb_image)
            self.img_list.append(img_as_tensor)

    def __getitem__(self, index):
        """Get an image tensor from the dataset with on-the-fly transformation.

        Args:
            index (int): The index of the image tensor in the dataset.

        Returns:
            torch.Tensor: The image tensor with on-the-fly transformation.
        """
        img = self.img_list[index]
        lbls = float(self.label_list[index])
        # Normalize the labels
        lbls_norm = lbls #self.normalize_lbls(lbls)
        
#         img = tensor_to_img(img)
        img = self.on_the_fly_transform(img)
#         img = image_to_tensor(img)
        
        return img,lbls_norm
    def normalize_lbls(self,labels):
        return labels/len(self.classes)

    def denormalize_lbls(self,labels):
        lbls = labels*(len(self.classes))
        # Need to see what I need whetehr labels or just norm values
        return lbls

    def get_pil_image(self, index, with_random_augmentation=True,label=False):
        """Get a PIL image from the dataset with or without on-the-fly transformation.

        Args:
            index (int): The index of the PIL image in the dataset.
            with_random_augmentation (bool, optional): Whether to apply on-the-fly transformation. Default is True.

        Returns:
            PIL.Image: The PIL image with or without on-the-fly transformation.
        """
        if with_random_augmentation:
            return tensor_to_img(self.__getitem__(index)[0])
        return tensor_to_img(self.img_list[index])

    def __len__(self):
        return len(self.img_list)


dataset = ImageDataset(img_paths="./pokemon/*png")

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Note hat just grabbing an image from the dataset adds a random image augmentation.
# Generally speaking, this will be False. Try a few times.
torch.all(dataset[0][0] == dataset[0][0])  # torch.all test if all entries are True

"""### Inspecting the Dataset"""

# Visualize the data

# Show seven versions of the first three Pokemons. 
images_per_row = 7
fig, axes = plt.subplots(3, images_per_row, figsize=(45, 15))

# Plotting each image in a subplot
for i, ax in enumerate(axes.flat):
    # The leftmost Pokemon is without augmentation (with_random_augmentation is False).
    with_random_augmentation = False if i % images_per_row == 0 else True
    # print(i//images_per_row)
    # print(with_random_augmentation)
    ax.imshow(dataset.get_pil_image(i//images_per_row, with_random_augmentation=with_random_augmentation))
    ax.axis('off')

plt.savefig('dataset_summary.png', bbox_inches='tight')
plt.show()

"""Next, we visualize the frequency of each RGB value in the image dataset (before augmentation). After the forward process, this should look like three standard normal distributions."""

# Create giant tensor with all images.
img_num = len(dataset.img_list)
tensor_with_all_images = torch.zeros((img_num, CHANNELS, IMG_SIZE, IMG_SIZE))
for i, img in enumerate(dataset.img_list):
    tensor_with_all_images[i,:,:,:] = img

# Save the pixel values of all images in 1D tensor for each channel. 
pixels_red = tensor_with_all_images[:,0,:,:].flatten().numpy()
pixels_green = tensor_with_all_images[:,1,:,:].flatten().numpy()
pixels_blue = tensor_with_all_images[:,2,:,:].flatten().numpy()

ax = sns.kdeplot(pixels_red, color="red", alpha=0.5, ls="--", lw=3)
sns.kdeplot(pixels_green, color="green", ax=ax, alpha=0.5, lw=3)
sns.kdeplot(pixels_blue, color="blue", ax=ax, alpha=0.5, ls=":", lw=3)
plt.title("Color distribution of transformed original images")
plt.savefig('color_distribution_original_images.png', dpi=300, bbox_inches='tight')

"""# Visualizing the Forward Process"""

def extract_single(a, t, x_shape):
    t = torch.tensor(t)
    batch_size = t.shape
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(DEVICE)

def one_step_forward(img, t):
    
    beta_start = BETA_START
    beta_end = BETA_END
    beta = torch.linspace(beta_start, beta_end, steps = TIMESTEPS,dtype = torch.float) 

    # Using alpha and alpha bar after reparametrization
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, axis=0)
    sqrt_alphas_bar = torch.sqrt(alpha_bar)
    sqrt_one_minus_alphas_bar = torch.sqrt(1. - alpha_bar)
 
    # 2. Calculate mean and variance
    sigma = extract_single(sqrt_one_minus_alphas_bar, t, img.shape)
    mu = extract_single(sqrt_alphas_bar, t, img.shape)  
    return img.to(DEVICE) * mu.to(DEVICE)  + sigma.to(DEVICE) * torch.randn_like(img, device=DEVICE)

def create_forward_animation(dataset):
    img = dataset.img_list[0].to(DEVICE)
    img_list = list()
    # if TIMESTEPS is too large, you can use a subset
    img_list.append(tensor_to_img(img))

    for t in range(TIMESTEPS):
        img = one_step_forward(img, t=t)
        img_list.append(tensor_to_img(img)) 
    return img_list
  
img_list = create_forward_animation(dataset)

# the .gif file can get pretty large
imageio.mimsave("forward_animate.gif", img_list, fps=10) 
# we can also show the images inline
show_from_image_list(img_list, filename = 'forward_grid.png')

forward_img_list = list()
for n,img in enumerate(dataset.img_list):
    # forward_img_list.append(image_to_tensor(img))
    img1 = one_step_forward(img.to(DEVICE), t=TIMESTEPS-1)

    forward_img_list.append(img1)

# Create giant tensor with all images.
img_num = len(forward_img_list)
tensor_with_all_images = torch.zeros((img_num, CHANNELS, IMG_SIZE, IMG_SIZE))
for i, img in enumerate(forward_img_list):
    tensor_with_all_images[i,:,:,:] = img

# Save the pixel values of all images in 1D tensor for each channel. 
pixels_red_f = tensor_with_all_images[:,0,:,:].flatten().numpy()
pixels_green_f = tensor_with_all_images[:,1,:,:].flatten().numpy()
pixels_blue_f = tensor_with_all_images[:,2,:,:].flatten().numpy()

ax = sns.kdeplot(pixels_red_f, color="red", alpha=0.5, ls="--", lw=3)
sns.kdeplot(pixels_green_f, color="green", ax=ax, alpha=0.5, lw=3)
sns.kdeplot(pixels_blue_f, color="blue", ax=ax, alpha=0.5, ls=":", lw=3)
plt.title("Color distribution of images in the last step of forward process")
plt.savefig('color_distribution_noisy_images.png', dpi=100, bbox_inches='tight')

"""### Positional Encoding
Position encoding for the time input. # Considered only Sinusoidal and Linear Embedding.
Shamefully stolen from https://github.com/tanelp/tiny-diffusion/blob/master/positional_embeddings.py
"""

class SinusoidalEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: torch.Tensor):
        device = x.device
        x = x * self.scale
        half_size = self.size // 2
        emb = torch.log(torch.Tensor([10000.0])) / (half_size - 1)
        emb = torch.exp(-emb.to(device) * torch.arange(half_size,device=device))
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb

    def __len__(self):
        return self.size


class LinearEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: torch.Tensor):
        x = x / self.size * self.scale
        return x.unsqueeze(-1)

    def __len__(self):
        return 1

class PositionalEmbedding(nn.Module):
    def __init__(self, size: int, type: str, **kwargs):
        super().__init__()

        if type == "sinusoidal":
            self.layer = SinusoidalEmbedding(size, **kwargs)
        elif type == "linear":
            self.layer = LinearEmbedding(size, **kwargs)
        else:
            raise ValueError(f"Unknown positional embedding type: {type}")

    def forward(self, x: torch.Tensor):
        return self.layer(x)

"""### Layer Normalization type
Selet type of normalization. In paper they have used group normalization in convolutional and attention layers
"""

def norm_type(norm, num_channels, num_groups):
    if norm == "in":
        return nn.InstanceNorm2d(num_channels, affine=True)
    elif norm == "bn":
        return nn.BatchNorm2d(num_channels)
    elif norm == "gn":
        return nn.GroupNorm(num_groups, num_channels)
    elif norm is None:
        return nn.Identity()
    else:
        raise ValueError("unknown normalization type")

"""Self Attention is used as per DDPM paper, with various normalization"""

class AttentionBlock(nn.Module):
    """Applies QKV self-attention with a residual connection.
    
    Input:
        x: tensor of shape (N, in_channels, H, W)
        norm (string or None): which normalization to use (instance, group, batch, or none). Default: "gn"
        num_groups (int): number of groups used in group normalization. Default: 32
    Output:
        tensor of shape (N, in_channels, H, W)
    Args:
        in_channels (int): number of input channels
    """
    def __init__(self, in_channels, norm="gn", num_groups=32):
        super().__init__()
        
        self.in_channels = in_channels
        self.norm = norm_type(norm, in_channels, num_groups)
        self.to_qkv = nn.Conv2d(in_channels, in_channels * 3, 1)
        self.to_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = torch.split(self.to_qkv(self.norm(x)), self.in_channels, dim=1)

        q = q.permute(0, 2, 3, 1).view(b, h * w, c)
        k = k.view(b, c, h * w)
        v = v.permute(0, 2, 3, 1).view(b, h * w, c)

        dot_products = torch.bmm(q, k) * (c ** (-0.5))
        assert dot_products.shape == (b, h * w, h * w)

        attention = torch.softmax(dot_products, dim=-1)
        out = torch.bmm(attention, v)
        assert out.shape == (b, h * w, c)
        out = out.view(b, h, w, c).permute(0, 3, 1, 2)

        return self.to_out(out) + x

"""DownSample and Upsample the layers"""

class Downsample(nn.Module):
    """Downsamples a given tensor by a factor of 2. Uses strided convolution. Assumes even height and width.
    Input:
        x: tensor of shape (N, in_channels, H, W)
        time_emb: ignored
        y: ignored
    Output:
        tensor of shape (N, in_channels, H // 2, W // 2)
    Args:
        in_channels (int): number of input channels
    """

    def __init__(self, in_channels):
        super().__init__()

        self.downsample = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)
    
    def forward(self, x, time_emb, y):
        if x.shape[2] % 2 == 1:
            raise ValueError("downsampling tensor height should be even")
        if x.shape[3] % 2 == 1:
            raise ValueError("downsampling tensor width should be even")

        return self.downsample(x)

class Upsample(nn.Module):
    """Upsamples a given tensor by a factor of 2. Uses resize convolution to avoid checkerboard artifacts.
    Input:
        x: tensor of shape (N, in_channels, H, W)
        time_emb: ignored
        y: ignored
    Output:
        tensor of shape (N, in_channels, H * 2, W * 2)
    Args:
        in_channels (int): number of input channels
    """

    def __init__(self, in_channels):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
        )
    
    def forward(self, x, time_emb, y):
        return self.upsample(x)

class ResidualBlock(nn.Module):
    """Applies two conv blocks with resudual connection. Adds time and class conditioning by adding bias after first convolution.
    Input:
        x: tensor of shape (N, in_channels, H, W)
        time_emb: time embedding tensor of shape (N, time_emb_dim) or None if the block doesn't use time conditioning
        y: classes tensor of shape (N) or None if the block doesn't use class conditioning
    Output:
        tensor of shape (N, out_channels, H, W)
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        time_emb_dim (int or None): time embedding dimension or None if the block doesn't use time conditioning. Default: None
        num_classes (int or None): number of classes or None if the block doesn't use class conditioning. Default: None
        activation (function): activation function. Default: torch.nn.functional.relu
        norm (string or None): which normalization to use (instance, group, batch, or none). Default: "gn"
        num_groups (int): number of groups used in group normalization. Default: 32
        use_attention (bool): if True applies AttentionBlock to the output. Default: False
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        dropout,
        time_emb_dim=None,
        num_classes=None,
        activation=F.relu,
        norm="gn",
        num_groups=32,
        use_attention=False,
    ):
        super().__init__()

        self.activation = activation

        self.norm_1 = norm_type(norm, in_channels, num_groups)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.norm_2 = norm_type(norm, out_channels, num_groups)
        self.conv_2 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        self.time_bias = nn.Linear(time_emb_dim, out_channels) if time_emb_dim is not None else None
        self.class_bias = nn.Embedding(num_classes, out_channels) if num_classes is not None else None

        self.residual_connection = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.attention = nn.Identity() if not use_attention else AttentionBlock(out_channels, norm, num_groups)
    
    def forward(self, x, time_emb=None, y=None):
        out = self.activation(self.norm_1(x))
        out = self.conv_1(out)

        if self.time_bias is not None:
            if time_emb is None:
                raise ValueError("time conditioning was specified but time_emb is not passed")
            out += self.time_bias(self.activation(time_emb))[:, :, None, None]

        if self.class_bias is not None:
            if y is None:
                raise ValueError("class conditioning was specified but y is not passed")

            out += self.class_bias(y)[:, :, None, None]

        out = self.activation(self.norm_2(out))
        out = self.conv_2(out) + self.residual_connection(x)
        out = self.attention(out)

        return out

class Network_UNet(nn.Module):
    
    """ 
    Input:
         x:tensor of shape (N, in_channels, H, W)
         time_emb: time embedding tensor of shape (N, t_dim) 
         y: class tensor of shape (N) for applying class conditioning. This is where we will condition with classes of the various categories
    
    Args:
        img_channels (int): number of image channels
        base_channels (int): number of base channels (after first convolution)
        channel_mults (tuple): tuple of channel multiplers. Default: (1, 2, 4, 8)
        t_dim (int or None): time embedding dimension or None if the block doesn't use time conditioning. Default: None
        time_emb_scale (float): linear scale to be applied to timesteps. Default: 1.0
        num_classes (int or None): number of classes or None if the block doesn't use class conditioning. Default: None
        activation (function): activation function. Default: torch.nn.functional.relu
        dropout (float): dropout rate at the end of each residual block
        attention_resolutions (tuple): list of relative resolutions at which to apply attention. Default: ()
        norm (string or None): which normalization to use (instance, group, batch, or none). Default: "gn"
        num_groups (int): number of groups used in group normalization. Default: 32
        initial_pad (int): initial padding applied to image. Should be used if height or width is not a power of 2. Default: 0
    """
    def __init__(self, img_channels = 3, 
                 base_channels = 64,
                 t_dim= 128,
                 time_emb_type= "sinusoidal", 
                 chn_mult = (1,2,4,8),
                 activation_fn=F.relu,
                 norm = 'gn',
                 num_groups = 32,
                 attention_resolutions=(),
                 num_classes = None,
                 num_res_blocks=2,
                 dropout = 0.1,
                 resnet_feat=None
                 ):
        super().__init__()


        self.time_embed_mlp = nn.Sequential( 
            PositionalEmbedding(t_dim, time_emb_type),
            nn.Linear(t_dim,t_dim),
            nn.ReLU()
            ) if t_dim is not None else None
        
        self.num_classes = num_classes
        self.first_conv = nn.Conv2d(img_channels,base_channels,3, padding=1)

        self.activation = activation_fn
        # Down Sample
        self.down_blocks = nn.ModuleList()
        # Updample
        self.up_blocks = nn.ModuleList()

        chns = [base_channels]
        now_chns = base_channels
        
        for i,mult in enumerate(chn_mult):
            out_chns = mult * base_channels
            for _ in range(num_res_blocks):

                self.down_blocks.append(ResidualBlock(
                        now_chns,
                        out_chns,
                        dropout,
                        time_emb_dim=t_dim,
                        num_classes=num_classes,
                        activation=activation_fn,
                        norm=norm,
                        num_groups=num_groups,
                        use_attention=i in attention_resolutions,
                ))
                now_chns = out_chns
                chns.append(now_chns)
            if i != len(chn_mult) - 1:
                self.down_blocks.append(Downsample(now_chns))
                chns.append(now_chns)

        self.mid_blocks = nn.ModuleList([
            ResidualBlock(now_chns, now_chns, dropout, time_emb_dim=t_dim, num_classes = num_classes,
                          activation = activation_fn, norm=norm, num_groups = num_groups, use_attention = True),
            ResidualBlock(now_chns, now_chns, dropout, time_emb_dim=t_dim, num_classes = num_classes,
                          activation = activation_fn, norm=norm, num_groups = num_groups, use_attention = False)
        ]) 
        for i,mult in reversed((list(enumerate(chn_mult)))):
            out_chns = base_channels * mult

            for _ in range(num_res_blocks + 1):
                self.up_blocks.append(ResidualBlock(
                    chns.pop() + now_chns,
                    out_chns,
                    dropout,
                    time_emb_dim=t_dim,
                    num_classes=num_classes,
                    activation=activation_fn,
                    norm=norm,
                    num_groups = num_groups,
                    use_attention = i in attention_resolutions
                ))
                now_chns = out_chns
            if i != 0:
                self.up_blocks.append(Upsample(now_chns))
        assert len(chns) == 0

        self.out_norm = norm_type(norm,base_channels,num_groups)
        self.tail_conv = nn.Conv2d(base_channels, img_channels,3,padding=1)

    def forward(self, x, t=None, y=None,im_feat=None):
        
        if self.time_embed_mlp is not None:
            if t is None:
                raise ValueError("time conditioning is not none but time is not passed")
            
            t_emb = self.time_embed_mlp(t)
        else:
            t_emb = None
        
        if self.num_classes is not None and y is None:
            raise ValueError("class conditioning was specified but y is not passed")
        

        x = self.first_conv(x)
        h_skips = [x]  # the skip connection s that bypass the features from down_blocks to up_blocks

        for layer in self.down_blocks:
            x = layer(x,t_emb,y)
            h_skips.append(x)

        for layer in self.mid_blocks:
            x = layer(x,t_emb,y)

        for layer in self.up_blocks:
            if isinstance(layer, ResidualBlock):
                x = torch.cat([x, h_skips.pop()], dim=1)
            x = layer(x, t_emb, y)            
        
        x = self.activation(self.out_norm(x))
        x = self.tail_conv(x)

        assert len(h_skips) == 0
        return x


# Various schedulers

def linear_beta_schedule(beta_start, beta_end, timesteps):
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(beta_start, beta_end,timesteps):
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(beta_start, beta_end, timesteps):
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

def scheduler_type(beta_start, beta_end, timesteps,type_sc='linear'):
    if type_sc == "linear":
        return linear_beta_schedule(beta_start, beta_end, timesteps)
    elif type_sc == "quadratic":
        return quadratic_beta_schedule(beta_start, beta_end, timesteps)
    elif type_sc == "sigmoid":
        return sigmoid_beta_schedule(beta_start, beta_end, timesteps)
    else:
        raise ValueError("unknown scheduler type")

"""### Diffusion Process"""

class DiffusionProcess(nn.Module):
    def __init__(self, model, T,beta_start=0.0001,beta_end=0.05, sch_type = 'linear'):
        super().__init__()
        
        self.T_steps = T
        
        self.betas = scheduler_type(beta_start, beta_end, self.T_steps,type_sc=sch_type)

        # Using alpha and alpha bar after reparametrization
        # Remeber to check the size of alphas and alpha_bar and alpha_bar_prev
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, axis=0)
        self.alphas_bar_prev = F.pad(self.alphas_bar[:-1], (1, 0), value=1.0)

        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for forward diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1. - self.alphas_bar)

        # calculations for reverse process posterior q(x_{t-1} | x_t, x_0) (eqn - 7)
        self.posterior_variance = self.betas * (1.- self.alphas_bar_prev) / (1. - self.alphas_bar)    

    # Importantly, extract function will allow us to extract the appropriate t index for a batch of indices.
    def extract(self,a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    
    def forward_process(self,img, t, noise= None):
        # Forward Diffusion
        # Make the alpha same dimension as that of the image
        if noise is None:
            noise = torch.randn_like(img, device=DEVICE)

        sqrt_alphas_bar_t = self.extract(self.sqrt_alphas_bar, t, img.shape)  # mu
        sqrt_one_minus_alphas_bar_t = self.extract(self.sqrt_one_minus_alphas_bar, t, img.shape)   # sigma 

        return img * sqrt_alphas_bar_t  + sqrt_one_minus_alphas_bar_t * noise 
        
    # x_{t-1} in line 4 of algorithm 2
    @torch.no_grad()
    def previous_sample(self,model,x,t,t_index,y=None,im_feat=None):
        betas_t = self.extract(self.betas,t,x.shape)
        sqrt_one_minus_alphas_bar_t = self.extract(self.sqrt_one_minus_alphas_bar, t, x.shape)
        sqrt_one_minus_alphas_t = self.extract(self.sqrt_recip_alphas,t, x.shape)
        # Equation 11 papeer. The mean of the model
        mean_mu_theta = sqrt_one_minus_alphas_t * (x - betas_t * model(x, t, y,im_feat) / sqrt_one_minus_alphas_bar_t)
        if t_index == 0:
            return mean_mu_theta
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape) # sigma_t^2 in line 4 of algorithm 2
            noise = torch.randn_like(x)
            # Algorithm 2 of Line 4
            return mean_mu_theta + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def run_sample(self,model, image_size, batch_size=16, channels=3, device=None,lbl=None,img_feat=None):

        shape = (batch_size, channels, image_size, image_size)

        batch_size = shape[0]

        if lbl is not None and batch_size != len(lbl):
            raise ValueError("sample batch size is different from length of given labels")
        if img_feat is not None and batch_size != len(img_feat):
            raise ValueError("sample batch size different from length of img_features")
        img = torch.randn(shape,device=device)            
        imgs = [] # for storing the reconstructed imgs over the timesteps

        for i in tqdm(reversed(range(0, self.T_steps)), desc='sampling loop time step', total=self.T_steps):
            img = self.previous_sample(model, img, torch.full((batch_size,), i, device=device, dtype=torch.long), i,lbl,img_feat)
            if i%TIME_DIV==0:
                imgs.append(img.cpu())
        return imgs

"""### Model , Optimizer and Diffusion Process initialization"""

#  Training 

# define the number of classes based on pokemon type for conditioning.
# This is not required for bonus so define on the top of the notebook appropriately
if CLASS_COND is not None:
    no_classes = len(dataset.classes)  # yes using
else:
    no_classes = None    # not using

# Initialize model correctly
model = Network_UNet(
    img_channels = CHANNELS,
    base_channels = 64,
    t_dim= TIME_DIMENSION,
    time_emb_type= TIME_EMBEDDING_TYPE, 
    chn_mult = (1,2,4,8),
    activation_fn=F.relu,
    norm = 'gn',
    num_groups = NUM_GROUPS,
    num_classes = no_classes,
    num_res_blocks=2,
    dropout = 0.1,
    resnet_feat = FEAT_COND
)
model.to(DEVICE)

optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

# Initialize the diffusion process. This includes the forward and the reverse process
diff_prs = DiffusionProcess(model, TIMESTEPS,beta_start=BETA_START,beta_end=BETA_END, sch_type = NOISE_SCHEDULER_TYPE)

"""### Loss functions used in Diffusion
Here you can add different losses used in various diffusion literature. We have used only mse loss 
"""

def diffusion_losses(denoise_model, x_start, t, y=None, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = diff_prs.forward_process(x_start, t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t, y,im_feat = x_start)
    
    if loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise) # check the gerrits way
    else:
        raise NotImplementedError()

    return loss

"""### Training loop
We are training for 2000 epochs and after every 10 epochs we are saving the model predictions and the FID (Frechet Inception Distance) score
"""
def main():
    
    last_best_loss = 100  # for saving the best weights
    fid = FID(feature=64) # # for computing the FID score
    for epoch in range(EPOCHS):
        running_loss = 0.0
        model.train()
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            imgs_bt_, lbl_bt_ = batch
            imgs_bt, lbl_bt = imgs_bt_.to(DEVICE), lbl_bt_.long().to(DEVICE)

            t = torch.randint(0, TIMESTEPS, (imgs_bt.shape[0],), device=DEVICE).long()

            loss = diffusion_losses(model, imgs_bt, t, loss_type=LOSS_TYPE,y = lbl_bt)

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() 
      
        tr_loss = running_loss/len(dataloader)
        
        # save intermediate generated images after every 10 epoch 
        # for checking the performance of the model
        if epoch%50==0:
            model.eval()
            with torch.no_grad():            
                bt_sze = imgs_bt.shape[0]

                lbl_cnd = lbl_bt if CLASS_COND is not None else None
                imgs_feat = imgs_bt if FEAT_COND is not None else None
                
                samples = diff_prs.run_sample(model, image_size=IMG_SIZE, batch_size=bt_sze, channels=CHANNELS,img_feat = imgs_feat,lbl=lbl_cnd,device=DEVICE)

                all_images = torch.stack(samples,dim=1)
                all_images = all_images.permute(1,0,2,3,4)# made it time_steps x batch x ch x h x w 

                re = imgs_bt.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8)
                fa = all_images[-1,:,:,:,:].mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8)
                fid.update(re, real=True)
                fid.update(fa, real=False)
                print('FID Score: {}'.format(fid.compute()))    
                
                all_images = all_images[:,:5,:,:,:].permute(1,0,2,3,4).contiguous().view(-1,CHANNELS,IMG_SIZE,IMG_SIZE) 

                all_images = (all_images + 1) * 0.5
                save_image(all_images, str(results_folder / f'sample-{epoch}.png'), nrow = 10)

        print("Epoch:{} | Loss:{}".format(epoch,tr_loss))
        if tr_loss<=last_best_loss:
            print("Saving model...")
            torch.save(model.state_dict(), f"{outdir}/model_best.pth")
            last_best_loss=tr_loss
        else:
            torch.save(model.state_dict(), f"{outdir}/model_last.pth")

    """### Generation of pokemons

    Show the types of Pokemon and create a tensor of specific pokemon types based on your choice
    """

    type_pok = 17
    if CLASS_COND is not None:
        pokemon_type = {}
        for n,cls in enumerate(dataset.classes):
            pokemon_type['{}'.format(cls)]=n
        print(pokemon_type)
        
        lbl_tensor = torch.tensor(type_pok)
        lbl_ten= lbl_tensor.repeat(BATCH_SIZE,1).to(DEVICE).squeeze()

    # sample images
    print('Inferencing checking')
    tesx = 'best'

    fid = FID(feature=192) # # for computing the FID score
    saved_weights = torch.load(os.path.join(outdir,'model_{}.pth'.format(tesx)), map_location=torch.device('cuda'))

    no_classes = len(dataset.classes)


    model.load_state_dict(saved_weights) 
    model.eval()
    img_ten = next(iter(dataloader))[0].to(DEVICE)

    lbl_ten1 = lbl_ten if CLASS_COND is not None else None
    img_ten1 = img_ten if FEAT_COND is not None else None
    samples = diff_prs.run_sample(model, image_size=IMG_SIZE, batch_size=BATCH_SIZE, channels=CHANNELS,\
                                  device=DEVICE,lbl=lbl_ten1,img_feat=img_ten1)# lbl=lbl_ten,img_feat=img_ten

    all_pred_images = torch.stack(samples,dim=1)
    all_pred_images = all_pred_images.permute(1,0,2,3,4) # made it time_steps x batch x ch x h x w 

    re = img_ten.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8)
    fa = all_pred_images[-1,:,:,:,:].mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8)
    fid.update(re, real=True)
    fid.update(fa, real=False)
    print('FID Score: {}'.format(fid.compute()))   

    all_test = []
    for i in range(all_pred_images.shape[0]):
        all_pred =  torchvision.utils.make_grid(all_pred_images[i],nrows=8).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        all_test.append(all_pred)
    imageio.mimsave(os.path.join(save_img_dir,"backward_animate_{}_{}.gif".format(experiment_name,tesx)), all_test, fps=5)

    print("Saving Final image...")

    imageio.imwrite(os.path.join(save_img_dir,"pred_image_{}_{}.png".format(experiment_name,tesx)), all_test[-1])


if __name__ == '__main__':
    main()

