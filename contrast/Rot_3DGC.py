# follow FS-Net
import torch.nn as nn
import backbone.fs_net_repo.gcn3d_hs as gcn3d_hs
import torch
import torch.nn.functional as F
from absl import app
import sys
sys.path.append('..')
from config.config_contrast import get_config 

FLAGS = get_config()



class Pts_3DGC(nn.Module):
    def __init__(self):
        super(Pts_3DGC, self).__init__()
        self.neighbor_num = FLAGS.gcn_n_num
        self.support_num = FLAGS.gcn_sup_num

        # 3D convolution for point cloud
        self.conv_0 = gcn3d_hs.HSlayer_surface(kernel_num=128, support_num=self.support_num)
        self.conv_1 = gcn3d_hs.HS_layer(128, 128, support_num=self.support_num)
        self.pool_1 = gcn3d_hs.Pool_layer(pooling_rate=4, neighbor_num=4)
        self.conv_2 = gcn3d_hs.HS_layer(128, 256, support_num=self.support_num)
        self.conv_3 = gcn3d_hs.HS_layer(256, 256, support_num=self.support_num)
        self.pool_2 = gcn3d_hs.Pool_layer(pooling_rate=4, neighbor_num=4)
        self.conv_4 = gcn3d_hs.HS_layer(256, 512, support_num=self.support_num)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)

        

    def forward(self,
                vertices: "tensor (bs, vetice_num, 3)"
                ):
        """
        Return: (bs, vertice_num, class_num)
        """
        #  concate feature
        bs, vertice_num, _ = vertices.size()
        
        # bs x verticenum x 6

        # ss = time.time()
        fm_0 = F.relu(self.conv_0(vertices, self.neighbor_num), inplace=True)
        fm_1 = F.relu(self.bn1(self.conv_1(vertices, fm_0, self.neighbor_num).transpose(1, 2)).transpose(1, 2), inplace=True)
        v_pool_1, fm_pool_1 = self.pool_1(vertices, fm_1)
        fm_2 = F.relu(self.bn2(self.conv_2(v_pool_1, fm_pool_1, 
                                           min(self.neighbor_num, v_pool_1.shape[1] // 8)).transpose(1, 2)).transpose(1, 2), inplace=True)
        fm_3 = F.relu(self.bn3(self.conv_3(v_pool_1, fm_2, 
                                           min(self.neighbor_num, v_pool_1.shape[1] // 8)).transpose(1, 2)).transpose(1, 2), inplace=True)
        v_pool_2, fm_pool_2 = self.pool_2(v_pool_1, fm_3)
        fm_4 = self.conv_4(v_pool_2, fm_pool_2, min(self.neighbor_num, v_pool_2.shape[1] // 8))
        f_global = fm_4.max(1)[0]  # (bs, f)
        
        nearest_pool_1 = gcn3d_hs.get_nearest_index(vertices, v_pool_1)
        nearest_pool_2 = gcn3d_hs.get_nearest_index(vertices, v_pool_2)
        fm_2 = gcn3d_hs.indexing_neighbor_new(fm_2, nearest_pool_1).squeeze(2)
        fm_3 = gcn3d_hs.indexing_neighbor_new(fm_3, nearest_pool_1).squeeze(2)
        fm_4 = gcn3d_hs.indexing_neighbor_new(fm_4, nearest_pool_2).squeeze(2)
        #one_hot = one_hot.unsqueeze(1).repeat(1, vertice_num, 1)  # (bs, vertice_num, cat_one_hot)
        
        feat = torch.cat([fm_0, fm_1, fm_2, fm_3, fm_4], dim=2) # bs*1024*1280
        
        return f_global, feat


