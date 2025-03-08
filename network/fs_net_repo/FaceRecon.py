# Modified from FS-Net
import torch.nn as nn
import network.fs_net_repo.gcn3d as gcn3d
import torch
import torch.nn.functional as F
from absl import app
import absl.flags as flags

FLAGS = flags.FLAGS


class FaceRecon(nn.Module):
    def __init__(self):
        super(FaceRecon, self).__init__()
        self.neighbor_num = FLAGS.gcn_n_num
        self.support_num = FLAGS.gcn_sup_num

        self.recon_num = 3
        self.face_recon_num = FLAGS.face_recon_c

        
        # 16: total 6 categories, 256 is global feature
        self.clip_r_dim = FLAGS.clip_r_dim 
        self.clip_t_dim = FLAGS.clip_t_dim 
        
        if FLAGS.train:
            self.conv1d_block_R = nn.Sequential(
                nn.Conv1d(self.clip_r_dim + 6, 512, 1),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Conv1d(512, 512, 1),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Conv1d(512, 256, 1),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
            )

            self.recon_head_R = nn.Sequential(
                nn.Conv1d(256, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, self.recon_num, 1),
            )

            self.face_head_R = nn.Sequential(
                nn.Conv1d(1545, 1024, 1),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
                nn.Conv1d(1024, 512, 1),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Conv1d(512, 256, 1),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Conv1d(256, self.face_recon_num, 1),  # Relu or not?
            )
            
            self.conv1d_block_t = nn.Sequential(
                nn.Conv1d(self.clip_t_dim + 6, 512, 1),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Conv1d(512, 512, 1),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Conv1d(512, 256, 1),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
            )

            self.recon_head_t = nn.Sequential(
                nn.Conv1d(256, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, self.recon_num, 1),
            )

            self.face_head_t = nn.Sequential(
                nn.Conv1d(2185, 1024, 1),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
                nn.Conv1d(1024, 512, 1),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Conv1d(512, 256, 1),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Conv1d(256, self.face_recon_num, 1),  # Relu or not?
            )
    def forward(self,
                vertices: "tensor (bs, vetice_num, 3)",
                cat_id: "tensor (bs, 1)",
                clip_r_feat: "tensor (bs, vertice, 1280)", 
                clip_t_feat: "tensor (bs, vertice, 1920)"
                ):
        """
        Return: (bs, vertice_num, class_num)
        """
        if len(vertices.shape) == 2:
            vertice_num, _ = vertices.size()
        else:
            bs, vertice_num, _ = vertices.size()
        # cat_id to one-hot
        
        if cat_id.shape[0] == 1:
            obj_idh = cat_id.view(-1, 1).repeat(cat_id.shape[0], 1)
        else:
            obj_idh = cat_id.view(-1, 1)

        one_hot = torch.zeros(bs, FLAGS.obj_c).to(cat_id.device).scatter_(1, obj_idh.long(), 1)
        one_hot = one_hot.unsqueeze(1).repeat(1, vertice_num, 1)  # (bs, vertice_num, cat_one_hot)
        
        
        # f_global = torch.cat((clip_r_feat, clip_t_feat),dim=2) # (bs, vertice, 3200)
        
        # feat = torch.cat((f_global, one_hot), dim=2) # (bs, vertice, 3206)
        feat_r = torch.cat([clip_r_feat, one_hot], dim=2)
        feat_t = torch.cat([clip_t_feat, one_hot], dim=2)
        
        if FLAGS.train:
            feat_face_re_R = feat_r
            feat_face_re_t = feat_t
            # feat is the extracted per pixel level feature
            
            conv1d_input_R = feat_r.permute(0, 2, 1)  # (bs, fuse_ch, vertice_num)
            conv1d_out_R = self.conv1d_block_R(conv1d_input_R)
            
            conv1d_input_t = feat_t.permute(0, 2, 1)  # (bs, fuse_ch, vertice_num)
            conv1d_out_t = self.conv1d_block_t(conv1d_input_t)
            
            recon_R = self.recon_head_R(conv1d_out_R)
            recon_t = self.recon_head_t(conv1d_out_t)
            
            # average pooling for face prediction
            feat_face_in_R = torch.cat([feat_face_re_R.permute(0,2,1), conv1d_out_R, vertices.permute(0, 2, 1)], dim=1) # (bs, 3459, 1024)
            feat_face_in_t = torch.cat([feat_face_re_t.permute(0,2,1), conv1d_out_t, vertices.permute(0, 2, 1)], dim=1) # (bs, 3459, 1024)
            
            face_R = self.face_head_R(feat_face_in_R) 
            face_t = self.face_head_t(feat_face_in_t) 
            
            recon =  (recon_R + recon_t)/2.0
            face = (face_R + face_t) / 2.0
            return recon.permute(0, 2, 1), face.permute(0, 2, 1) 
        else:
            recon, face = None, None
            return recon, face 


