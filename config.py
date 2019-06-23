import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

im_size = 320
unknown_code = 128

# Training parameters
num_workers = 1  # for data-loading; right now, only 1 works with h5py
grad_clip = 5.  # clip gradients at an absolute value of
print_freq = 100  # print training/validation stats  every __ batches
checkpoint = None  # path to checkpoint, None if none

##############################################################
# Set your paths here

# path to provided foreground images
fg_path = 'data/fg/'

# path to provided alpha mattes
a_path = 'data/mask/'

# Path to background images (MSCOCO)
bg_path = 'data/bg/'

# Path to folder where you want the composited images to go
out_path = 'data/merged/'

##############################################################
