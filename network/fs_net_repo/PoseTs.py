import torch.nn as nn
import torch
import torch.nn.functional as F
import absl.flags as flags
from absl import app
from .Cross_Atten import CrossAttention
FLAGS = flags.FLAGS

# Point_center  encode the segmented point cloud
# one more conv layer compared to original paper

class Pose_Ts(nn.Module):
    def __init__(self):
        super(Pose_Ts, self).__init__()
        self.f = FLAGS.feat_c_ts
        self.k = FLAGS.Ts_c
        self.clip_t_dim = FLAGS.clip_t_dim
        
        self.first_convs = nn.Sequential(
            nn.Conv1d(self.f + 3, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Conv1d(1024, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            
        )
        self.second_convs = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, self.k),
        )
        
    def forward(self, clip_feat_t=None, use_clip=False, use_clip_global=False):
        
        x = self.first_convs(clip_feat_t) 
        
        x = torch.max(x, 2, keepdim=True)[0].squeeze() 
        
        re = self.second_convs(x)
        # re = re.squeeze(2)
        # re = torch.max(re, 2, keepdim=True)[0].squeeze()
        re = re.contiguous()
        xt = re[:, 0:3]
        xs = re[:, 3:6]
        return xt, xs