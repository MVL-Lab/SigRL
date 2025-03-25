from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
from models.SDA import SemanticDecoupling
from models.VFR import DecouplingSemantic
import torch.nn.functional as F
from models.graphs.gatv2.experiment import GATv2
import dhg
from dhg.models import GAT
from dhg import Graph

class CLIPVIT(nn.Module):

    def __init__(self, args, clip_model, embed_dim=768):
        super().__init__()

        self.final_dim = 512
        self.global_only = False
        
        self.conv1 = clip_model.visual.conv1
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.transformer = clip_model.visual.transformer
        self.ln_post = clip_model.visual.ln_post
        self.clipzero = False

        self.use_clip_proj = False

        if not self.use_clip_proj:
            self.projection = nn.Sequential(OrderedDict([
                    ('fc1', nn.Linear(embed_dim, self.final_dim)),
                    ('act', nn.Tanh()),
                    ('fc2', nn.Linear(self.final_dim, self.final_dim))],)
            )

        self.projection_dist = clip_model.visual.proj
        self.topk = args.topk

        self.intermediaDim = 512
        self.imageFeatureDim = 512
        self.wordFeatureDim = 512
        self.outputDim = 512
        self.SemanticDecoupling = SemanticDecoupling(self.imageFeatureDim, self.wordFeatureDim, intermediaDim=self.intermediaDim)
        self.DecouplingSemantic = DecouplingSemantic(self.imageFeatureDim, 1024, intermediaDim=self.intermediaDim)
        # #ist
        self.relu = nn.ReLU(inplace=True)
        self.GATv2_1 = GATv2(in_features = 512, n_hidden = 128, out_features = 512, n_heads = 4,dropout= 0.3, share_weights = True, is_concat = True)

    
    #ist
    def getConcatIndex(self, classNum):
        res = [[], []]
        for index in range(classNum-1):
            res[0] += [index for i in range(classNum-index-1)]
            res[1] += [i for i in range(index+1, classNum)]
        return res
    
    def forward_features(self, x):
        
        x = self.conv1(x)  
        x = x.reshape(x.shape[0], x.shape[1], -1)  
        x = x.permute(0, 2, 1) 
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)


        return x

    # 4 CoOp label_embed
    def forward(self, x, label_embed, norm_pred=True):        
        adj_mat = np.ones((label_embed.size()[0],label_embed.size()[0]))
        adj_mat = torch.from_numpy(adj_mat).unsqueeze(-1).to(x.device)
        label_embed_graph = self.GATv2_1(label_embed,adj_mat)


        x = self.forward_features(x)
        dist_feat = x[:, 0] @ self.projection_dist

        # For Global Head Only Ablation 只处理cls部分
        if self.global_only:
            score = dist_feat @ label_embed.t()
            if norm_pred:
                score = score / score.norm(dim=-1, keepdim=True)
            return score, x[:, 1:], dist_feat

        # Default global+local
        else:
            pred_feat = x[:, 1:] @ self.projection_dist
            semanticFeature = self.SemanticDecoupling(pred_feat, label_embed)
            
            # [BatchSize * classNum * Dim]
            output = (semanticFeature.permute(0, 2, 1) @ label_embed)/(semanticFeature.norm()*label_embed.norm()) # [bs, Dim, Dim]
            output = self.relu(output)
            output = F.normalize(output, dim=-1)
            output = semanticFeature @ output

            label_embed_graph = torch.unsqueeze(label_embed_graph,0)
            label_embed_graph = label_embed_graph.repeat(output.shape[0],1,1)
            output = torch.cat((output,label_embed_graph),dim=2)
            SD_label = self.DecouplingSemantic(pred_feat, output)
            
            
           
            score_SD = torch.topk(SD_label @ label_embed.t(),k=self.topk, dim=1)[0].mean(dim=1)

            score2 = dist_feat @ label_embed.t()


            if norm_pred:
                # score1 = score1 / score1.norm(dim=-1, keepdim=True)
                score_SD = score_SD / score_SD.norm(dim=-1, keepdim=True)
                score2 = score2 / score2.norm(dim=-1, keepdim=True)
            
            score = (score_SD + score2) / 2 

            return score, pred_feat, dist_feat

    def encode_img(self, x):
        # import pdb; pdb.set_trace
        x = self.forward_features(x)
        if self.clipzero:
            x = x @ self.proj
            return x[:, 1:, :], x[:, 0, :]
        else:
            pred_feat = x[:, 1:] @ self.projection_dist

            dist_feat = x[:, 0] @ self.projection_dist
            return pred_feat, dist_feat