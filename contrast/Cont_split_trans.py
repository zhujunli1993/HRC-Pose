import torch
from torch import nn
import torch.nn.functional as F
import pytorch3d
import sys
sys.path.append('..')
from .rnc_loss import RnCLoss_trans_mix,RnCLoss_trans_mug_mix,RnCLoss_trans_nonSym_mix
from config.config_contrast import get_config  
from .Rot_3DGC import Pts_3DGC

CFG = get_config()

class Projection(nn.Module):
    
    def __init__(
        self,
        pts_embedding=CFG.pts_embedding
        ):
        super(Projection, self).__init__()
        self.projection_dim = pts_embedding
        self.w1 = nn.Linear(pts_embedding, pts_embedding, bias=False)
        self.bn1 = nn.BatchNorm1d(pts_embedding)
        self.relu = nn.ReLU()
        self.w2 = nn.Linear(pts_embedding, self.projection_dim, bias=False)
        self.bn2 = nn.BatchNorm1d(self.projection_dim, affine=False)
    
    def forward(self, embedding):
        
        return self.bn2(self.w2(self.relu(self.bn1(self.w1(embedding)))))
        
class Model_Trans_all(nn.Module):
    def __init__(
        self,
        k1=CFG.k1,
        k2=CFG.k2,
        temperature=CFG.temperature,
        pts_embedding=CFG.pts_embedding,
        pose_embedding=256,
    ):
        super(Model_Trans_all, self).__init__()
        ''' encode point clouds '''
        
        self.pts_encoder = Pts_3DGC()

        self.project_head = Projection(1283)
        self.temperature = temperature
        
        self.clrk = Class_Rank(temperature=self.temperature,base_temperature=self.temperature)
    
    
       
    def forward(self, batch, umap=False, for_test=False, for_decoder=False):

        
        bs = batch['pts'].shape[0]
        mean_point = batch['pts'].mean(dim=1, keepdim=True) 
        _, pts_1_features_pp = self.pts_encoder(batch['pts']-mean_point)
        pts_1_features_pp = torch.cat([pts_1_features_pp, batch['pts']-mean_point], dim=2)
        
        pts_1_features = pts_1_features_pp.max(1)[0] # (bs, 1283)

        pts_1_features = self.project_head(pts_1_features) #bs*N*3
        
        
        if torch.all(torch.isnan(pts_1_features))==False and torch.all(torch.isinf(pts_1_features))==False:
            if not for_decoder and not umap:
                # Getting point cloud and gt pose Features
                gt_pose = batch['zero_mean_gt_pose']
                gt_R = gt_pose[:, :9].reshape(bs,3,3)
                gt_green, gt_red = get_gt_v(gt_R)
                sym = batch['sym']
                gt_t = batch['gt_pose'][:, 9:] - batch['pts'].mean(dim=1)
                labels = batch['id']
                trans_loss = self.clrk(pts_1_features, labels, gt_green, gt_red, gt_t, sym)
                
                return trans_loss, pts_1_features_pp.permute(0,2,1)
            
            
            if for_decoder and not umap:
                return pts_1_features_pp.permute(0,2,1)
            
            if umap:
                # Getting point cloud and gt pose Features
                gt_pose = batch['zero_mean_gt_pose']
                gt_R = gt_pose[:, :9].reshape(bs,3,3)
                gt_green, gt_red = get_gt_v(gt_R)
                sym = batch['sym']
                gt_t = batch['gt_pose'][:, 9:]  - batch['pts'].mean(dim=1)
                labels = batch['id']
                trans_loss = self.clrk(pts_1_features, labels, gt_green, gt_red, gt_t, sym)
                return trans_loss, pts_1_features, gt_t
        else:
            import pdb;pdb.set_trace()
            return None                
        
        
   
                
class Class_Rank(nn.Module):
    def __init__(self, temperature=2,
                 base_temperature=2, layer_penalty=None, loss_type='hmce'):
        super(Class_Rank, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        if not layer_penalty:
            self.layer_penalty = self.pow_2
        else:
            self.layer_penalty = layer_penalty
        # self.sup_con_loss = SupConLoss(temperature=self.temperature, contrast_mode='all', base_temperature=self.temperature, feature_sim='l2')
        self.loss_type = loss_type
        
        self.rnc_loss_nonSym = RnCLoss_trans_nonSym_mix(temperature=self.temperature,soft_lambda=0.800, label_diff='l1', feature_sim='l2')
        self.rnc_loss = RnCLoss_trans_mix(temperature=self.temperature, soft_lambda=0.800,label_diff='l1', feature_sim='l2')
        self.rnc_loss_mug = RnCLoss_trans_mug_mix(temperature=self.temperature, soft_lambda=0.800, label_diff='l1', feature_sim='l2')
    
    def pow_2(self, value):
        return torch.pow(2, value)

    def forward(self, features, labels, gt_green, gt_red, gt_trans, sym):
        device = features.device
        bs = labels.shape[0]

        trans_layer_loss = torch.tensor(0.0).to(device)
        all_ids = torch.unique(labels)
        
        for i in all_ids:
            
            ind = torch.where(labels == i)[0]

            sym_ind = (sym[ind, 0] == 0).nonzero(as_tuple=True)[0] # find non-sym objects
            feat_id, green_id, red_id, gt_trans_id = features[ind], gt_green[ind], gt_red[ind], gt_trans[ind]
            
            if i == 5:
                trans_layer_loss += self.rnc_loss_mug(feat_id, green_id, red_id, gt_trans_id, sym[ind])
                
            else:
                if len(sym_ind) == 0: # sym obj
                    trans_layer_loss += self.rnc_loss(feat_id, green_id, gt_trans_id)
                    
                else:
                    trans_layer_loss += self.rnc_loss_nonSym(feat_id, green_id, red_id, gt_trans_id)
                    
        return trans_layer_loss / len(all_ids)
