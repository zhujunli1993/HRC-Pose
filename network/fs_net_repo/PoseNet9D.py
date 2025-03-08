import torch
import torch.nn as nn
import absl.flags as flags
from absl import app
import numpy as np
import torch.nn.functional as F
from contrast.Cont_split_trans import Model_Trans_all as CLIPModel_trans
from contrast.Cont_split_rot import Model_Rot_all as CLIPModel_rot
from network.fs_net_repo.PoseR import Rot_red, Rot_green
from network.fs_net_repo.PoseTs import Pose_Ts
from network.fs_net_repo.FaceRecon import FaceRecon

FLAGS = flags.FLAGS

class PoseNet9D(nn.Module):
    def __init__(self):
        super(PoseNet9D, self).__init__()
        # Used the fsnet rot_green and rot_red directly
        self.rot_green = Rot_green() 
        self.rot_red = Rot_red()
        self.rot_green = Rot_green()
        self.rot_red = Rot_red()
        
        self.clip_r_func = CLIPModel_rot() 
        if FLAGS.pretrained_clip_rot_model_path:
            self.clip_r_func.load_state_dict(torch.load(FLAGS.pretrained_clip_rot_model_path))
        self.clip_t_func = CLIPModel_trans() 
        if FLAGS.pretrained_clip_t_model_path:
            self.clip_t_func.load_state_dict(torch.load(FLAGS.pretrained_clip_t_model_path))
        
        if not FLAGS.train:
            # Freeze clip model parameters
            for param in self.clip_t_func.parameters():
                param.requires_grad = False     
            for param in self.clip_r_func.parameters():
                param.requires_grad = False 
        self.face_recon = FaceRecon()
        
        self.ts = Pose_Ts()
        

    def forward(self, batch, points, obj_id, use_clip=True, use_clip_global=False, use_clip_nonLinear=False, use_clip_atte=False):
        bs, p_num = points.shape[0], points.shape[1]
        
        if FLAGS.train:
            clip_loss_r, clip_r_feat = self.clip_r_func(batch)
            clip_loss_t, clip_t_feat = self.clip_t_func(batch)
        else:
            clip_r_feat = self.clip_r_func(batch,umap=False, for_decoder=True)
            clip_t_feat = self.clip_t_func(batch,umap=False, for_decoder=True)
            
        recon, face = self.face_recon(points - points.mean(dim=1, keepdim=True), obj_id,clip_r_feat, clip_t_feat)
        
        if FLAGS.train:
            recon = recon + points.mean(dim=1, keepdim=True)
            # handle face
            face_normal = face[:, :, :18].view(bs, p_num, 6, 3)  # normal
            face_normal = face_normal / torch.norm(face_normal, dim=-1, keepdim=True)  # bs x nunm x 6 x 3
            face_dis = face[:, :, 18:24]  # bs x num x  6
            face_f = F.sigmoid(face[:, :, 24:])  # bs x num x 6
        else:
            face_normal, face_dis, face_f, recon = [None]*4
    
        #  rotation
        
        green_R_vec = self.rot_green( clip_r_feat, use_clip=True, use_clip_global=True)
        red_R_vec = self.rot_red( clip_r_feat, use_clip=True, use_clip_global=True)
        # normalization
        p_green_R = green_R_vec[:, 1:] / (torch.norm(green_R_vec[:, 1:], dim=1, keepdim=True) + 1e-6)
        p_red_R = red_R_vec[:, 1:] / (torch.norm(red_R_vec[:, 1:], dim=1, keepdim=True) + 1e-6)
        # sigmoid for confidence
        f_green_R = F.sigmoid(green_R_vec[:, 0])
        f_red_R = F.sigmoid(red_R_vec[:, 0])

        # translation and size
        
        clip_t_feat = torch.cat([clip_t_feat, points-points.mean(dim=1, keepdim=True)], dim=2) #(bs, 1024, 1923)
        T, s = self.ts(clip_t_feat.permute(0,2,1), use_clip=True, use_clip_global=True)
        Pred_T = T + points.mean(dim=1)  # bs x 3
        Pred_s = s  # this s is not the object size, it is the residual

        if FLAGS.train:
            return clip_loss_r, clip_loss_t, recon, face_normal, face_dis, face_f, p_green_R, p_red_R, f_green_R, f_red_R, Pred_T, Pred_s
        else:
            return recon, face_normal, face_dis, face_f, p_green_R, p_red_R, f_green_R, f_red_R, Pred_T, Pred_s


class PoseNet9D_save(nn.Module):
    def __init__(self):
        super(PoseNet9D_save, self).__init__()
        # Used the fsnet rot_green and rot_red directly
        self.rot_green = Rot_green() 
        self.rot_red = Rot_red()
        if FLAGS.use_clip_nonLinear == 1.0 and FLAGS.use_clip_atte==0.0:
            self.rot_green = Rot_green_nonLinear()
            self.rot_red = Rot_red_nonLinear()
        if FLAGS.use_clip_atte == 1.0:
            self.rot_green = Rot_green_atten()
            self.rot_red = Rot_red_atten()
            
        self.face_recon = FaceRecon()
        
        self.ts = Pose_Ts()
        if FLAGS.use_clip_nonLinear == 1.0:
            self.ts = Pose_Ts_nonLinear()
        if FLAGS.use_clip_atte == 1.0:
            self.ts = Pose_Ts_atten()

    def forward(self, points, obj_id, clip_r_feat, clip_t_feat, use_clip=True, use_clip_global=False, use_clip_nonLinear=False, use_clip_atte=False):
        bs, p_num = points.shape[0], points.shape[1]
        recon, face, feat, clip_r_feat_pixel, clip_t_feat_pixel = self.face_recon(points - points.mean(dim=1, keepdim=True), obj_id,
                                            clip_r_feat, clip_t_feat)

        if FLAGS.train:
            recon = recon + points.mean(dim=1, keepdim=True)
            # handle face
            face_normal = face[:, :, :18].view(bs, p_num, 6, 3)  # normal
            face_normal = face_normal / torch.norm(face_normal, dim=-1, keepdim=True)  # bs x nunm x 6 x 3
            face_dis = face[:, :, 18:24]  # bs x num x  6
            face_f = F.sigmoid(face[:, :, 24:])  # bs x num x 6
        else:
            face_normal, face_dis, face_f, recon = [None]*4
    
        #  rotation
        if use_clip:
            feat_and_clip_r = torch.cat([feat, clip_r_feat_pixel], dim=-1)
        else:
            feat_and_clip_r = feat
        if use_clip_global is True and use_clip_nonLinear is False:
            green_R_vec = self.rot_green(feat.permute(0, 2, 1), clip_r_feat, use_clip=True, use_clip_global=True)
            red_R_vec = self.rot_red(feat.permute(0, 2, 1), clip_r_feat, use_clip=True, use_clip_global=True)
        elif use_clip_global is False and use_clip_nonLinear is False:
            green_R_vec = self.rot_green(feat_and_clip_r.permute(0, 2, 1))  # b x 4
            red_R_vec = self.rot_red(feat_and_clip_r.permute(0, 2, 1))   # b x 4
        elif use_clip_global is False and use_clip_nonLinear is True:
            green_R_vec = self.rot_green(feat.permute(0, 2, 1), clip_r_feat, use_clip=True, use_clip_global=False)
            red_R_vec = self.rot_red(feat.permute(0, 2, 1), clip_r_feat, use_clip=True, use_clip_global=False)
        elif use_clip_global is True and use_clip_atte is True and use_clip_nonLinear is False:
            green_R_vec = self.rot_green(feat.permute(0, 2, 1), clip_r_feat, use_clip=True, use_clip_global=True)
            red_R_vec = self.rot_red(feat.permute(0, 2, 1), clip_r_feat, use_clip=True, use_clip_global=True)
        # normalization
        p_green_R = green_R_vec[:, 1:] / (torch.norm(green_R_vec[:, 1:], dim=1, keepdim=True) + 1e-6)
        p_red_R = red_R_vec[:, 1:] / (torch.norm(red_R_vec[:, 1:], dim=1, keepdim=True) + 1e-6)
        # sigmoid for confidence
        f_green_R = F.sigmoid(green_R_vec[:, 0])
        f_red_R = F.sigmoid(red_R_vec[:, 0])

        # translation and size
        if use_clip:
            feat_and_clip_t = torch.cat([feat, clip_t_feat_pixel], dim=-1)
        else:
            feat_and_clip_t = feat
        if use_clip_global is True and use_clip_nonLinear is False:    
            feat_for_ts = torch.cat([feat, points-points.mean(dim=1, keepdim=True)], dim=2)
            T, s = self.ts(feat_for_ts.permute(0, 2, 1), clip_t_feat, use_clip=True, use_clip_global=True)
        elif use_clip_global is False and use_clip_nonLinear is False:
            feat_for_ts = torch.cat([feat_and_clip_t, points-points.mean(dim=1, keepdim=True)], dim=2)
            T, s = self.ts(feat_for_ts.permute(0, 2, 1))
        elif use_clip_global is False and use_clip_nonLinear is True:
            feat_for_ts = torch.cat([feat, points-points.mean(dim=1, keepdim=True)], dim=2)
            T, s = self.ts(feat_for_ts.permute(0, 2, 1), clip_t_feat, use_clip=True, use_clip_global=False)
        elif use_clip_global is True and use_clip_atte is True and use_clip_nonLinear is False:
            feat_for_ts = torch.cat([feat, points-points.mean(dim=1, keepdim=True)], dim=2)
            T, s = self.ts(feat_for_ts.permute(0, 2, 1), clip_t_feat, use_clip=True, use_clip_global=True)
        Pred_T = T + points.mean(dim=1)  # bs x 3
        Pred_s = s  # this s is not the object size, it is the residual
        feat = torch.max(feat,dim=1)
        
        return recon, face_normal, face_dis, face_f, p_green_R, p_red_R, f_green_R, f_red_R, Pred_T, Pred_s, feat.values

def main(argv):
    classifier_seg3D = PoseNet9D()

    points = torch.rand(2, 1000, 3)
    import numpy as np
    obj_idh = torch.ones((2, 1))
    obj_idh[1, 0] = 5
    '''
    if obj_idh.shape[0] == 1:
        obj_idh = obj_idh.view(-1, 1).repeat(points.shape[0], 1)
    else:
        obj_idh = obj_idh.view(-1, 1)

    one_hot = torch.zeros(points.shape[0], 6).scatter_(1, obj_idh.cpu().long(), 1)
    '''
    recon, f_n, f_d, f_f, r1, r2, c1, c2, t, s = classifier_seg3D(points, obj_idh)
    t = 1



if __name__ == "__main__":
    print(1)
    from config.config import *
    app.run(main)





