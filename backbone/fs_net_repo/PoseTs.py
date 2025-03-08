import torch.nn as nn
import torch
import torch.nn.functional as F

from absl import app
import sys
sys.path.append('..')
from configs.config import get_config 

FLAGS = get_config()

# Point_center  encode the segmented point cloud
# one more conv layer compared to original paper
class Pose_Ts_global(nn.Module):
    def __init__(self, input_c):
        super(Pose_Ts_global, self).__init__()
        self.f = input_c
        self.k = FLAGS.Ts_c

        self.layer1 = torch.nn.Linear(self.f, 512)
        self.layer2 = torch.nn.Linear(512, 1024)
        self.layer3 = torch.nn.Linear(1024, 512)
        self.layer4 = torch.nn.Linear(512, 256)
        self.layer5 = torch.nn.Linear(256, 64)
        self.layer6 = torch.nn.Linear(64, self.k)
        
        self.ln1 = nn.LayerNorm(512)
        self.ln2 = nn.LayerNorm(1024)
        self.ln3 = nn.LayerNorm(512)
        self.ln4 = nn.LayerNorm(256)
        self.ln5 = nn.LayerNorm(64)
        

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.ln1(self.layer1(x)))
        x = self.relu2(self.ln2(self.layer2(x)))
        x = self.relu3(self.ln3(self.layer3(x)))
        x = self.relu4(self.ln4(self.layer4(x)))
        x = self.relu5(self.ln5(self.layer5(x)))
        x = self.layer6(x)
        x = x.contiguous()
        return x   
class Pose_Ts(nn.Module):
    def __init__(self):
        super(Pose_Ts, self).__init__()
        self.f = FLAGS.feat_c_ts
        self.k = FLAGS.Ts_c

        self.conv1 = torch.nn.Conv1d(self.f, 1024, 1)

        self.conv2 = torch.nn.Conv1d(1024, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, self.k, 1)
        self.drop1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = torch.max(x, 2, keepdim=True)[0]

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.drop1(x)
        x = self.conv4(x)

        x = x.squeeze(2)
        x = x.contiguous()
        xt = x[:, 0:3]
        xs = x[:, 3:6]
        return xt, xs

def main(argv):
    feature = torch.rand(3, 3, 1000)
    obj_id = torch.randint(low=0, high=15, size=[3, 1])
    net = Pose_Ts()
    out = net(feature, obj_id)
    t = 1

if __name__ == "__main__":
    app.run(main)
