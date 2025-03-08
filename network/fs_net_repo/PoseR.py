import torch.nn as nn
import torch
import torch.nn.functional as F
import absl.flags as flags
from absl import app
from config.config import *
from .Cross_Atten import CrossAttention
FLAGS = flags.FLAGS


class Rot_green(nn.Module):
    def __init__(self):
        super(Rot_green, self).__init__()
        self.f = FLAGS.feat_c_R  
        self.k = FLAGS.R_c
        self.clip_r_dim = FLAGS.clip_r_dim
        
        
        self.first_convs = nn.Sequential(
            nn.Conv1d(self.f, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Conv1d(1024, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            
        )
        
        self.second_convs = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(256, self.k, 1),
        )
    def forward(self,  clip_feat_r=None, use_clip=False,use_clip_global=False):
        
        # fuse_feat = torch.cat((x, clip_feat_r),dim=2) # (bs, 1024, 2566)
        
        x = self.first_convs(clip_feat_r.permute(0, 2, 1)) # (bs, 256, 1024)
        
        x = torch.max(x, 2, keepdim=True)[0] 
        
        re = self.second_convs(x)
        re = re.squeeze()
        
        re = re.contiguous()

        return re


class Rot_red(nn.Module):
    def __init__(self):
        super(Rot_red, self).__init__()
        self.f = FLAGS.feat_c_R  
        self.k = FLAGS.R_c
        self.clip_r_dim = FLAGS.clip_r_dim
        
        # self.mlp_r = nn.Sequential(
        #     nn.Conv1d(self.clip_r_dim, 512,1),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(512, 512,1),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(512, 512,1),
        # )
        self.first_convs = nn.Sequential(
            nn.Conv1d(self.f, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Conv1d(1024, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            
        )
        # self.cross = CrossAttention(dim=256, heads=2)
        # self.conv1 = torch.nn.Conv1d(self.f + 6, 1024, 1)
        # self.conv2 = torch.nn.Conv1d(1024, 512, 1)
        # self.conv3 = torch.nn.Conv1d(512, 256, 1)
        # self.conv4 = torch.nn.Conv1d(256, self.k, 1)
        # self.drop1 = nn.Dropout(0.2)
        # self.bn1 = nn.BatchNorm1d(1024)
        # self.bn2 = nn.BatchNorm1d(512)
        # self.bn3 = nn.BatchNorm1d(256)
        self.second_convs = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(256, self.k, 1),
        )
    def forward(self,  clip_feat_r=None, use_clip=False,use_clip_global=False):
        
        # fuse_feat = torch.cat((x, clip_feat_r),dim=2) # (bs, 1024, 2566)
        
        x = self.first_convs(clip_feat_r.permute(0, 2, 1)) # (bs, 256, 1024)
        
        x = torch.max(x, 2, keepdim=True)[0] 
        
        re = self.second_convs(x)
        re = re.squeeze()
        
        re = re.contiguous()

        return re