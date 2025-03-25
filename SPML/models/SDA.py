import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticDecoupling(nn.Module):

    def __init__(self, imgFeatureDim, wordFeatureDim, intermediaDim=1024):
        
        super(SemanticDecoupling, self).__init__()

        self.imgFeatureDim = imgFeatureDim
        self.wordFeatureDim = wordFeatureDim
        self.intermediaDim = intermediaDim

        #feature_map  bs*196*512
        #label_emb  classnum1006*512
        self.fc3 = nn.Linear(self.intermediaDim, self.intermediaDim)

    def forward(self, imgFeaturemap, wordFeatures, visualize=False):
        '''
        Shape of imgFeaturemap : (BatchSize, Channel, imgSize, imgSize) bs*196*512
        Shape of wordFeatures : (classNum, wordFeatureDim) classnum1006*512
        '''
       
        classNum = wordFeatures.shape[0] 
        bs = imgFeaturemap.size()[0]
        imgSize = imgFeaturemap.size()[1]
        dim = imgFeaturemap.size()[2]
        imgFeature = imgFeaturemap.contiguous().view(bs * imgSize, -1)                                             # (BatchSize * imgSize) * dim              
        imgFeature = imgFeature.view(bs * imgSize, 1, -1).repeat(1, classNum, 1)                                   # (BatchSize * imgSize) * classNum * dim

        wordFeature = wordFeatures.view(1, classNum, dim).repeat(bs * imgSize, 1, 1) # (BatchSize * imgSize) * classNum * dim
        
        feature = self.fc3(torch.tanh(imgFeature * wordFeature).view(-1, dim))                                       # (BatchSize * imgSize * classNum) * dim
        
        Coefficient = feature.view(bs, imgSize, classNum, dim)
        Coefficient = torch.transpose(torch.transpose(Coefficient, 2, 1),3 ,2) # BatchSize * classNum * dim* imgSize
        Coefficient = F.softmax(Coefficient, dim=3)

        Coefficient = torch.transpose(torch.transpose(Coefficient, 3, 1), 3, 2)                                                     # BatchSize * imgSize * classNum *dim

        imgFeature = imgFeature.view(bs, imgSize, classNum, dim)                                                                            # BatchSize * imgSize * classNum * dim
        featuremapWithCoefficient = imgFeature * Coefficient
        semanticFeature = torch.sum(featuremapWithCoefficient, 1)                        
        return semanticFeature

