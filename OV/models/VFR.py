import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class DecouplingSemantic(nn.Module):

    def __init__(self, imgFeatureDim, wordFeatureDim, intermediaDim=1024):
        
        # super(SemanticDecoupling, self).__init__()
        super().__init__()

        self.tokenDim = imgFeatureDim
        self.labelGraphDim = wordFeatureDim
        self.intermediaDim = intermediaDim

        self.fc1 = nn.Linear(self.labelGraphDim, self.intermediaDim, bias=False)
        self.fc3 = nn.Linear(self.intermediaDim, self.intermediaDim)

    def forward(self, tokenFeaturemap, labelGraphfeatures, visualize=False):
        '''
        Shape of tokenFeaturemap : (BatchSize, imgSize, dim) bs*196*512
        Shape of labelGraphfeatures : (bs,classNum, dim) 
        '''
        
        classNum = labelGraphfeatures.shape[1] 
        bs = tokenFeaturemap.shape[0]
        imgSize = tokenFeaturemap.shape[1]
        dim = tokenFeaturemap.shape[2]

        tokenFeaturemap = tokenFeaturemap.contiguous().view(bs * imgSize, -1)                                             # (BatchSize * imgSize) * dim              
        tokenFeaturemap = tokenFeaturemap.view(bs * imgSize, 1, -1).repeat(1, classNum, 1)                                   # (BatchSize * imgSize) * classNum * dim
        tokenFeaturemap = tokenFeaturemap.view(bs, imgSize, classNum, dim)                                                    # BatchSize * imgSize * classNum * dim
        # print(tokenFeaturemap.shape)

        labelGraphfeatures = self.fc1(labelGraphfeatures)                                                                 # BatchSize * imgSize * dim  1024->512
        labelGraphfeatures = labelGraphfeatures.view(1, bs, classNum, dim).repeat(imgSize, 1 , 1, 1) # imgSize * BatchSize * classNum * dim
        labelGraphfeatures = torch.transpose(labelGraphfeatures,1,0)
        # [bs,]
        # print(labelGraphfeatures.shape)

        feature = self.fc3(torch.tanh(tokenFeaturemap * labelGraphfeatures).view(-1, dim))                                       # (BatchSize * imgSize * classNum) * dim
        # print(feature.shape)
        Coefficient = feature.view(bs, imgSize, classNum, dim) 
        Coefficient = torch.transpose(torch.transpose(Coefficient, 2, 1),3 ,2) # BatchSize * classNum * dim* imgSize
        Coefficient = F.softmax(Coefficient, dim=3)

        Coefficient = torch.transpose(torch.transpose(Coefficient, 3, 1), 3, 2)                                                     # BatchSize * imgSize * classNum *dim

        # imgFeature = imgFeature.view(bs, imgSize, classNum, dim)                                                                            # BatchSize * imgSize * classNum * dim
        featuremapWithCoefficient = tokenFeaturemap * Coefficient
        semanticFeature = torch.sum(featuremapWithCoefficient, 2)           
        return semanticFeature

