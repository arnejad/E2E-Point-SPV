import torch
from torch import nn
import torch.nn.functional as F
import math


######### MODELS DEFINITIONS
class SightedRecognizer(nn.Module):
    def __init__(self, Model, numOfLayers):
        super(SightedRecognizer, self).__init__()
        self.features = nn.Sequential(*list(Model.children())[:numOfLayers])
        
    def forward(self, x):
        x = self.features(x)
        return x


class BlindRecognizer_Feat(nn.Module):
    def __init__(self, Model):
        super(BlindRecognizer_Feat, self).__init__()
        # self.features = nn.Sequential(*list(Model.children())[:numOfLayers])
        self.feature = Model
        self.channelAdj = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1)
    def forward(self, x):
        # x = x.type(torch.LongTensor)
        x = self.channelAdj(x)
        x = self.feature(x)
        # x = self.softmax(x)
        return x


class BlindRecognizer_Classifier(nn.Module):
    def __init__(self, Model, numOfLayers, numOfOutputs):
        super(BlindRecognizer_Classifier, self).__init__()

        # self.classifierModel = nn.Sequential(*list(Model.children())[numOfLayers:])
        self.classifierModel = Model
        self.classChooser = nn.Linear(1000, numOfOutputs)
        # self.softmax = torch.nn.Softmax()
        
    def forward(self, x):
        x = self.classifierModel(x)
        x = self.classChooser(x)
        # x = self.softmax(x)
        return x

class Representer(torch.nn.Module):

    def __init__(self, arryaOut=False):
        super(Representer, self).__init__()

        self.arryaOut = arryaOut

        self.seq1 = nn.Sequential(nn.Conv2d(3,8,3,1,1, padding_mode='replicate'),
                                   nn.BatchNorm2d(8),
                                   nn.LeakyReLU(inplace=True))
        self.seq2 = nn.Sequential(nn.Conv2d(8,16,3,1,1, padding_mode='replicate'),
                                   nn.BatchNorm2d(16),
                                   nn.LeakyReLU(inplace=True), nn.MaxPool2d(2))
        self.seq3 = nn.Sequential(nn.Conv2d(16,32,3,1,1, padding_mode='replicate'),
                                   nn.BatchNorm2d(32),
                                   nn.LeakyReLU(inplace=True), nn.MaxPool2d(2))
        self.seq4 = nn.Sequential(nn.Conv2d(32,64,3,1,1, padding_mode='replicate'),
                                   nn.BatchNorm2d(64),
                                #    nn.LeakyReLU(inplace=True), nn.MaxPool2d(2))
                                nn.LeakyReLU(inplace=True))
        
        self.resBlk1 = ResidualBlock(64, resample_out=None)
        self.resBlk2 = ResidualBlock(64, resample_out=None)
        self.resBlk3 = ResidualBlock(64, resample_out=None)
        self.resBlk4 = ResidualBlock(64, resample_out=None)

        self.seq5 = nn.Sequential(nn.Conv2d(64,32,3,1,1, padding_mode='replicate'),
                                   nn.BatchNorm2d(32),
                                   nn.LeakyReLU(inplace=True), nn.MaxPool2d(2))
        
        self.seq6 = nn.Sequential(nn.Conv2d(32,16,3,1,1, padding_mode='replicate'),
                                   nn.BatchNorm2d(16),
                                   nn.LeakyReLU(inplace=True))
        
        self.seq7 = nn.Sequential(nn.Conv2d(16,8,3,1,1, padding_mode='replicate'),
                                   nn.BatchNorm2d(8),
                                   nn.LeakyReLU(inplace=True))
        
        self.conv8 = nn.Conv2d(8,3,3,1,1, padding_mode='replicate')
        self.conv9 = nn.Conv2d(3,1,3,1,1, padding_mode='replicate')
                                   
        self.lin1 = nn.Linear(in_features=3072, out_features=1024)
        self.lin2 = nn.Linear(in_features=1024, out_features=650)
        self.flat = nn.Flatten()
        self.sig = torch.nn.Sigmoid()
        self.tan = torch.nn.Tanh()
        # self.softmax = torch.nn.Softmax()

    def forward(self, x):
        # x = self.act1(self.conv1(x))
        x = self.seq1(x)
        # x = self.act2(self.conv2(x))
        x = self.seq2(x)
        x = self.seq3(x)
        x = self.seq4(x)
        x = self.resBlk1(x)
        x = self.resBlk2(x)
        x = self.resBlk3(x)
        x = self.resBlk4(x)
        
        x = self.seq5(x)
        x = self.seq6(x)
        x = self.seq7(x)
        x = self.conv8(x)
        # x = self.sig(self.act6(self.conv6(x)))

        if not self.arryaOut:
            x = self.conv9(x)
            
        else:
            x = self.flat(x)
            x = self.lin1(x)
            x = self.lin2(x)


        x = self.tan(x)
        x = x + torch.sign(x).detach() - x.detach()
        x = .5*(x+1)
        
        return x

class ResidualBlock(nn.Module):
    def __init__(self, n_channels, stride=1, resample_out=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, n_channels,kernel_size=3, stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_channels, n_channels,kernel_size=3, stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(n_channels)
        self.resample_out = resample_out
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        if self.resample_out:
            out = self.resample_out(out)
        return out
    


#### SIMULATOR MODEL FROM EXPERIMENT 4 of Jaap de Ruyter van Steveninck, 2022
class Simulator_exp4(object):
    """ Modular phosphene simulator that is used in experiment 4. Requires a predefined phosphene mapping. e.g. Tensor of 650 X 256 X 256 where 650 is the number of phosphenes and 256 X 256 is the resolution of the output image."""
    def __init__(self,pMap=None, device='cpu',pMap_from_file='/home/burkuc/viseon_a/training_configuration/phosphene_map_exp4.pt'):
        # Phospene mapping (should be of shape: n_phosphenes, res_x, res_y)
        if pMap is not None:
            self.pMap = pMap
        else:
            #self.pMap = torch.load(pMap_from_file, map_location=torch.device('cpu'))
            self.pMap = torch.load(pMap_from_file, map_location=torch.device(device))


        self.n_phosphenes = self.pMap.shape[0]
    
    def __call__(self,stim):
        return torch.einsum('ij, jkl -> ikl', stim, self.pMap).unsqueeze(dim=1) 

    def get_center_of_phosphenes(self):
        pMap = torch.nn.functional.interpolate(self.pMap.unsqueeze(dim=1),size=(128,128))  #650,1,128,128
        pLocs = pMap.view(self.n_phosphenes,-1).argmax(dim=-1) #650
        self.plocs = pLocs // 128, pLocs % 128 # y and x coordinates of the center of each phosphene
        return pLocs
    


    
class E2E_PhospheneSimulator_jaap(nn.Module):
    """ Uses three steps to convert  the stimulation vectors to phosphene representation:
    1. Resizes the feature map (default: 32x32) to SVP template (256x256)
    2. Uses pMask to sample the phosphene locations from the SVP activation template
    2. Performs convolution with gaussian kernel for realistic phosphene simulations
    """
    def __init__(self,pMask,scale_factor=8, sigma=1.5,kernel_size=11, intensity=15, device=torch.device('cuda:0')):
        super(E2E_PhospheneSimulator_jaap, self).__init__()
        
        # Device
        self.device = device
        
        # Phosphene grid
        self.pMask = pMask.to(self.device)
        self.up = nn.Upsample(mode="nearest",scale_factor=scale_factor)
        self.gaussian = self.get_gaussian_layer(kernel_size=kernel_size, sigma=sigma, channels=1)
        self.intensity = intensity 
    
    def get_gaussian_layer(self, kernel_size, sigma, channels):
        """non-trainable Gaussian filter layer for more realistic phosphene simulation"""

        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1)/2.
        variance = sigma**2.

        # Calculate the 2-dimensional gaussian kernel
        gaussian_kernel = (1./(2.*math.pi*variance)) *\
                          torch.exp(
                              -torch.sum((xy_grid - mean)**2., dim=-1) /\
                              (2*variance)
                          )

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                    kernel_size=kernel_size, groups=channels, bias=False)

        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False

        return gaussian_filter    

    def forward(self, stimulation):
        
        # Phosphene simulation
        phosphenes = self.up(stimulation)*self.pMask
        phosphenes = self.gaussian(F.pad(phosphenes, (5,5,5,5), mode='constant', value=0)) 
        return self.intensity*phosphenes    
    

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x
    

def divideModel(resNetModel2, FEAT_LAYERS_N):

    modules = list(resNetModel2.children())[:FEAT_LAYERS_N]
    blindUnit_feat_pre = nn.Sequential(*modules)

    modules2 = list(resNetModel2.children())[FEAT_LAYERS_N:-1]
    blindUnit_classifier_pre = nn.Sequential(*[*modules2, Flatten(), list(resNetModel2.children())[-1]])

    return blindUnit_feat_pre, blindUnit_classifier_pre