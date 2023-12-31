import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import torch.nn.functional as F

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


class SgnetFeatureExtractor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SgnetFeatureExtractor, self).__init__()
        self.embbed_size = output_dim
        self.box_embed = nn.Sequential(nn.Linear(input_dim, self.embbed_size), 
                                        nn.ReLU()) 
    def forward(self, inputs):
        box_input = inputs
        embedded_box_input= self.box_embed(box_input)

        return embedded_box_input