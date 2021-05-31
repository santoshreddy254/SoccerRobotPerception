import torch
from torch import nn


detection_loss = nn.MSELoss()
segmentation_loss = nn.CrossEntropyLoss()

def total_variation(images):
    "Adapted from tf.image.total_variation"
    "(https://www.tensorflow.org/api_docs/python/tf/image/total_variation) "

    pixel_dif1 = images[:, :, :, :-1] - images[:, :, :, 1:]
    pixel_dif2 = images[:, :, :-1, :] - images[:, :, 1:, :]

    # Sum for all axis. (None is an alias for all axis.)
    # sum_axis = None
    tot_var = (torch.sum(torch.abs(pixel_dif1)) + torch.sum(torch.abs(pixel_dif2)))
    return tot_var/images.shape[0]
