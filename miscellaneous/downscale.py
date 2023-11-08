"""
https://www.youtube.com/watch?v=AqscP7rc8_M</br>
https://www.youtube.com/watch?v=poY_nGzEEWM&t=55s
"""

from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15, 15))

img = Image.open('taylor.png')

# Original image (320x320)
initial_transforms = transforms.Compose([
    transforms.ToTensor(),
])
img_tensor = initial_transforms(img)
original_image = img_tensor.permute(1, 2, 0)
axes[0].imshow(original_image)
axes[0].axis('off')
axes[0].set_title("Original")

# Nearest neighbor interpolation (80x80)
nearest_resize_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((80, 80), interpolation=transforms.InterpolationMode.NEAREST, antialias=True)
])
img_tensor = nearest_resize_transforms(img)
nearest_neighbor_image = img_tensor.permute(1, 2, 0)
axes[1].imshow(nearest_neighbor_image)
axes[1].axis('off')
axes[1].set_title("Nearest neighbor")

# Bilinear interpolation (80x80)
bilinear_resize_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((80, 80), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
])
img_tensor = bilinear_resize_transforms(img)
bilinear_image = img_tensor.permute(1, 2, 0)
axes[2].imshow(bilinear_image)
axes[2].axis('off')
axes[2].set_title("Bilinear")

# Bicubic interpolation (80x80)
bicubic_resize_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((80, 80), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
])
img_tensor = bicubic_resize_transforms(img)
bicubic_image = img_tensor.permute(1, 2, 0)
axes[3].imshow(bicubic_image)
axes[3].axis('off')
axes[3].set_title("Bicubic")

# Show
plt.show()