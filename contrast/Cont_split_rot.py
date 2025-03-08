import torch
from torch import nn
import torch.nn.functional as F
import pytorch3d
import sys
sys.path.append('..')
from .rnc_loss import RnCLoss_rot_mug_mix, RnCLoss_rot_nonSym_mix,  RnCLoss_rot_mug_mix
from config.config_contrast import get_config 
from tools.training_utils import get_gt_v
from .Rot_3DGC import Pts_3DGC
CFG = get_config()
def unique(x, dim=None):
    """Unique elements of x and indices of those unique elements
    https://github.com/pytorch/pytorch/issues/36748#issuecomment-619514810

    e.g.
    unique(tensor([
        [1, 2, 3],
        [1, 2, 4],
        [1, 2, 3],
        [1, 2, 5]
    ]), dim=0)
    => (tensor([[1, 2, 3],
                [1, 2, 4],
                [1, 2, 5]]),
        tensor([0, 1, 3]))
    """
    unique, inverse = torch.unique(
        x, sorted=True, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype,
                        device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    return unique, inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
class FeatureSimilarity(nn.Module):
    def __init__(self, similarity_type='l2'):
        super(FeatureSimilarity, self).__init__()
        self.similarity_type = similarity_type

    def forward(self, features):
        # labels: [bs, feat_dim]
        # output: [bs, bs]
        if self.similarity_type == 'l2':
            return -(features[:, None, :] - features[None, :, :]).norm(2, dim=-1) # negative L2 norm
        elif self.similarity_type == 'cos':
            
            # make sure the last row is [0, 0, 0, 1]
            bs = features.shape[0]
            cos = nn.CosineSimilarity(dim=2, eps=1e-8)
            features_x1 = features.unsqueeze(1)  # bs*1*3
            features_x2 = features.unsqueeze(0)  # 1*bs*3
            
            return cos(features_x1, features_x2)
            
        else:
            raise ValueError(self.similarity_type)
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
        

class Model_Rot_all(nn.Module):
    def __init__(
        self,
        k1=CFG.k1,
        k2=CFG.k2,
        temperature=CFG.temperature,
        pts_embedding=CFG.pts_embedding,
        pose_embedding=256,
    ):
        super(Model_Rot_all, self).__init__()
        ''' encode point clouds '''
        self.pts_encoder = Pts_3DGC()

        self.project_head = Projection(512)
        self.temperature = temperature
        
        self.clrk = Class_Rank(temperature=self.temperature,base_temperature=self.temperature)
    
       
    def forward(self, batch, umap=False, for_decoder=False):
        
        bs = batch['zero_mean_pts_1'].shape[0]
        # Getting point cloud and gt pose Features
        
        pts_1_features = self.project_head(self.pts_encoder(batch['zero_mean_pts_1'])) #bs*N*3
        
        
        if torch.all(torch.isnan(pts_1_features))==False and torch.all(torch.isinf(pts_1_features))==False:
            if not for_decoder and not umap:
                gt_pose = batch['zero_mean_gt_pose']
                gt_R = gt_pose[:, :9].reshape(bs,3,3)
                gt_green, gt_red = get_gt_v(gt_R)

                labels = batch['id']
                sym = batch['sym']
                rot_loss = self.clrk(pts_1_features, labels, gt_green, gt_red, sym)
                return rot_loss

            if for_decoder and not umap:
                return pts_1_features
            
            if umap:
                gt_pose = batch['zero_mean_gt_pose']
                gt_R = gt_pose[:, :9].reshape(bs,3,3)
                gt_green, gt_red = get_gt_v(gt_R)

                labels = batch['id']
                sym = batch['sym']
                rot_loss = self.clrk(pts_1_features, labels, gt_green, gt_red, sym)
                return rot_loss, pts_1_features, gt_green, gt_red
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
        self.rnc_loss_nonSym = RnCLoss_rot_nonSym_mix(temperature=self.temperature,soft_lambda=0.800, label_diff='cos', feature_sim='l2')
        self.rnc_loss = RnCLoss_rot_mix(temperature=self.temperature, soft_lambda=0.800,label_diff='cos', feature_sim='l2')
        self.rnc_loss_mug = RnCLoss_rot_mug_mix(temperature=self.temperature, soft_lambda=0.800, label_diff='cos', feature_sim='l2')
    def pow_2(self, value):
        return torch.pow(2, value)

    def forward(self, features, labels, gt_green, gt_red, gt_trans, sym):
        device = features.device
        bs = labels.shape[0]

        rot_layer_loss = torch.tensor(0.0).to(device)
        all_ids = torch.unique(labels)
        
        for i in all_ids:
            
            ind = torch.where(labels == i)[0]

            sym_ind = (sym[ind, 0] == 0).nonzero(as_tuple=True)[0] # find non-sym objects
            feat_id, green_id, red_id, gt_trans_id = features[ind], gt_green[ind], gt_red[ind], gt_trans[ind]
            
            if i == 5:
                rot_layer_loss += self.rnc_loss_mug(feat_id, green_id, red_id, gt_trans_id, sym[ind])
                
            else:
                if len(sym_ind) == 0: # sym obj
                    rot_layer_loss += self.rnc_loss(feat_id, green_id, gt_trans_id)
                    
                else:
                    rot_layer_loss += self.rnc_loss_nonSym(feat_id, green_id, red_id, gt_trans_id)
                    
        return rot_layer_loss / len(all_ids)




