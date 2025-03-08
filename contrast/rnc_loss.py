
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch3d
import sys
sys.path.append('..')
from config.config_contrast import get_config 
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
            return - (features[:, None, :] - features[None, :, :]).norm(2, dim=-1) # negative L2 norm
        elif self.similarity_type == 'cos':
            
            # make sure the last row is [0, 0, 0, 1]
            bs = features.shape[0]
            cos = nn.CosineSimilarity(dim=2, eps=1e-8)
            features_x1 = features.unsqueeze(1)  # bs*1*3
            features_x2 = features.unsqueeze(0)  # 1*bs*3
            
            return cos(features_x1, features_x2)
            
        else:
            raise ValueError(self.similarity_type)


class LabelDifference_rot(nn.Module):
    def __init__(self, label_diff='l1'):
        super(LabelDifference_rot, self).__init__()
        self.label_diff = label_diff
    def div_rot(self, rot_mat):
        
        div = torch.pow((torch.linalg.det(rot_mat)), 1/3)
        div = div.unsqueeze(dim=-1)
        div = div.unsqueeze(dim=-1)
        rot_mat = rot_mat / div
        return rot_mat
    
    def rot_error_sym(self, rot):
        
        # make sure the last row is [0, 0, 0, 1]
        bs = rot.shape[0]
        
        rot_x1 = rot.unsqueeze(1)  # bs*1*3
        rot_x2 = rot.unsqueeze(0)  # 1*bs*3
    
        rot_losses = F.smooth_l1_loss(rot_x1, rot_x2, reduction='none',  beta=CFG.smooth_l1_beta) 

        return rot_losses
    def rot_cos_sym(self, rot):
        
        # make sure the last row is [0, 0, 0, 1]
        bs = rot.shape[0]
        cos = nn.CosineSimilarity(dim=2, eps=1e-8)
        rot_x1 = rot.unsqueeze(1)  # bs*1*3
        rot_x2 = rot.unsqueeze(0)  # 1*bs*3
        
        rot_losses = 1.0 - cos(rot_x1, rot_x2)
        return rot_losses

    
    def pose_error_sym(self, pose):
        
        # make sure the last row is [0, 0, 0, 1]
        bs = pose.shape[0]
        rot = pytorch3d.transforms.rotation_6d_to_matrix(pose[:,:6])
        rot = self.div_rot(rot)
        
        t = pose[:, 6:]
    
        # symmetric when rotating around y-axis
        
        y = torch.tensor([0., 1., 0.]).to(pose.device)
        y = torch.unsqueeze(y, dim=0).expand(bs, -1)
        y = torch.unsqueeze(y, dim=-1)
        
        rot_sym = torch.bmm(rot, y).squeeze()
        
        rot_x1 = rot_sym.unsqueeze(1)  # bs*1*3
        rot_x2 = rot_sym.unsqueeze(0)  # 1*bs*3
        
        cos_theta = (rot_x1 * rot_x2).sum(dim=2) / (torch.linalg.norm(rot_x1, dim=2) * torch.linalg.norm(rot_x2, dim=2))
        theta = torch.arccos(torch.clip(cos_theta, -1.0, 1.0)) 
        
        t_reshaped = torch.tile(t.unsqueeze(1), (1, t.shape[0], 1))
        shift = torch.linalg.norm(t_reshaped - t, dim=-1) 
        
          
        return theta, shift   
    
    def forward(self, rot):
        if self.label_diff == 'l1':
            rot_losses = self.rot_error_sym(rot)
        if self.label_diff == 'cos':
            rot_losses = self.rot_cos_sym(rot)
        
            
        if len(rot_losses.shape)==2:
            return rot_losses
        else:
            return rot_losses.mean(2)
        
class LabelDifference_trans(nn.Module):
    def __init__(self, label_diff='l1'):
        super(LabelDifference_trans, self).__init__()
        self.label_diff = label_diff
   
    
    def trans_error(self, trans):
        
        # make sure the last row is [0, 0, 0, 1]
        bs = trans.shape[0]
        
        trans_x1 = trans.unsqueeze(1)  # bs*1*3
        trans_x2 = trans.unsqueeze(0)  # 1*bs*3
    
        trans_losses = F.smooth_l1_loss(trans_x1, trans_x2, reduction='none',  beta=CFG.smooth_l1_beta) 

        return trans_losses
    
    def forward(self, trans):
        if self.label_diff == 'l1':
            trans_losses = self.trans_error(trans)
        if len(trans_losses.shape)==2:
            return trans_losses
        else:
            return trans_losses.mean(2)
class RnCLoss_rot_sym(nn.Module):
    def __init__(self, temperature=2, soft_lambda=0.500, label_diff='l1', feature_sim='l2'):
        super(RnCLoss_rot_sym, self).__init__()
        self.t = temperature
        self.label_diff_fn = LabelDifference_rot(label_diff)
        self.feature_sim_fn = FeatureSimilarity(feature_sim)
        self.soft_lambda = soft_lambda

    def forward(self, rot_feat, rot, complete=False):
        
        rot_diff = self.label_diff_fn(rot)

        rot_logits = self.feature_sim_fn(rot_feat).div(self.t)
        rot_logits_max, _ = torch.max(rot_logits, dim=1, keepdim=True)
        rot_logits -= rot_logits_max
        rot_exp_logits = rot_logits.exp()

        
        n = rot_logits.shape[0]  # n = 2bs

        # remove diagonal
        rot_logits = rot_logits.masked_select((1 - torch.eye(n).to(rot_logits.device)).bool()).view(n, n - 1)
        rot_exp_logits = rot_exp_logits.masked_select((1 - torch.eye(n).to(rot_logits.device)).bool()).view(n, n - 1)
       
        rot_diff = rot_diff.masked_select((1 - torch.eye(n).to(rot_logits.device)).bool()).view(n, n - 1)
        
        loss = 0.0
        
        rot_diff_expanded = rot_diff.unsqueeze(2).expand(-1, -1, n-1)
        rot_diff_expanded_columns = rot_diff_expanded.transpose(1, 2)
        
        
        rot_result_matrices = (rot_diff_expanded <= rot_diff_expanded_columns)
        
        rot_neg_mask_label =  rot_result_matrices.permute(1,0,2)
        
        
        eps = 1e-7
            
        pos_log_probs_rot = rot_logits.permute(1,0) - torch.log((rot_neg_mask_label * rot_exp_logits).sum(dim=-1)+eps)
        
        loss_rot = - (pos_log_probs_rot / (n * (n - 1))).sum()
        

        if not torch.isnan(loss_rot) and not torch.isinf(loss_rot): 
            return loss_rot
        else:
            return None  
class RnCLoss_trans_nonSym_mix(nn.Module):
    def __init__(self, temperature=2, soft_lambda=0.500, label_diff='l1', feature_sim='l2'):
        super(RnCLoss_trans_nonSym_mix, self).__init__()
        self.t = temperature
        
        self.label_diff_fn_rot = LabelDifference_rot('cos')
        self.label_diff_fn = LabelDifference_trans(label_diff)
        self.feature_sim_fn = FeatureSimilarity(feature_sim)
        self.soft_lambda = soft_lambda

    def forward(self, trans_feat, rot_green, rot_red, trans, complete=False):
        
        trans_diff = self.label_diff_fn(trans)
        
        rot_green_diff = self.label_diff_fn_rot(rot_green)
        rot_red_diff = self.label_diff_fn_rot(rot_red)
        
        rot_diff = rot_green_diff + rot_red_diff
        
        trans_logits = self.feature_sim_fn(trans_feat).div(self.t)
        trans_logits_max, _ = torch.max(trans_logits, dim=1, keepdim=True)
        trans_logits -= trans_logits_max
        trans_exp_logits = trans_logits.exp()

        n = trans_logits.shape[0]  # n = 2bs

        # remove diagonal
        trans_logits = trans_logits.masked_select((1 - torch.eye(n).to(trans_logits.device)).bool()).view(n, n - 1)
        trans_exp_logits = trans_exp_logits.masked_select((1 - torch.eye(n).to(trans_logits.device)).bool()).view(n, n - 1)
       
        trans_diff = trans_diff.masked_select((1 - torch.eye(n).to(trans_logits.device)).bool()).view(n, n - 1)
        rot_diff = rot_diff.masked_select((1 - torch.eye(n).to(trans_logits.device)).bool()).view(n, n - 1)
        
        loss = 0.0
        
        rot_diff_expanded = rot_diff.unsqueeze(2).expand(-1, -1, n-1)
        rot_diff_expanded_columns = rot_diff_expanded.transpose(1, 2)
        
        trans_diff_expanded = trans_diff.unsqueeze(2).expand(-1, -1, n-1)
        trans_diff_expanded_columns = trans_diff_expanded.transpose(1, 2)
        
        rot_result_matrices = (rot_diff_expanded <= rot_diff_expanded_columns)
        trans_result_matrices = (trans_diff_expanded <= trans_diff_expanded_columns)
        
        hard_neg_mask = (rot_result_matrices * trans_result_matrices).float() # rot_diff<=anchor_rot_diff AND trans_diff<=anchor_trans_diff
        
        trans_neg_mask_label =  trans_result_matrices.permute(1,0,2)
        hard_neg_mask = hard_neg_mask.permute(1,0,2)
        
        # hard_neg_mask = (rot_result_matrices * trans_result_matrices).float() # rot_diff<=anchor_rot_diff AND trans_diff<=anchor_trans_diff
        # soft_neg_mask = (rot_result_matrices + trans_result_matrices).float() # rot_diff<=anchor_rot_diff OR trans_diff<=anchor_trans_diff
        
        
        # hard_neg_mask = hard_neg_mask.permute(1,0,2)
        # soft_neg_mask = soft_neg_mask.permute(1,0,2)
        
        eps = 1e-7
            
        pos_log_probs_trans = trans_logits.permute(1,0) - torch.log((trans_neg_mask_label * trans_exp_logits).sum(dim=-1)+eps)
        pos_log_probs_hard = trans_logits.permute(1,0) - torch.log((hard_neg_mask * trans_exp_logits).sum(dim=-1)+eps)
        
        loss_hard = - (pos_log_probs_hard / (n * (n - 1))).sum()
        loss_trans = - (pos_log_probs_trans / (n * (n - 1))).sum()
        
        loss = loss_hard + self.soft_lambda * loss_trans
        
        if not torch.isnan(loss) and not torch.isinf(loss): 
            return loss
        else:
            return None 

class RnCLoss_trans_mug_mix(nn.Module):
    def __init__(self, temperature=2, soft_lambda=0.500, label_diff='l1', feature_sim='l2'):
        super(RnCLoss_trans_mug_mix, self).__init__()
        self.t = temperature
        self.label_diff_fn_rot = LabelDifference_rot('cos')
        self.label_diff_fn = LabelDifference_trans(label_diff)
        
        self.feature_sim_fn = FeatureSimilarity(feature_sim)
        self.soft_lambda = soft_lambda

    def forward(self, trans_feat, rot_green, rot_red, trans, sym, complete=False):
        trans_diff = self.label_diff_fn(trans)
        
        rot_green_diff = self.label_diff_fn_rot(rot_green)
        rot_red_diff = self.label_diff_fn_rot(rot_red)
        
        rot_diff = rot_green_diff + rot_red_diff
        sym_ind = (sym[:, 0] == 1).nonzero(as_tuple=True)[0] # all sym mug 
        rot_diff[sym_ind,:] = rot_green_diff[sym_ind,:] # for sym mug, only keep the green_vec difference between sym mug and other non-sym mug
        rot_diff[:,sym_ind] = rot_green_diff[:,sym_ind] # for sym mug, only keep the green_vec difference between sym mug and other non-sym mug
        
        trans_logits = self.feature_sim_fn(trans_feat).div(self.t)
        trans_logits_max, _ = torch.max(trans_logits, dim=1, keepdim=True)
        trans_logits -= trans_logits_max
        trans_exp_logits = trans_logits.exp()

        n = trans_logits.shape[0]  # n = 2bs

        # remove diagonal
        trans_logits = trans_logits.masked_select((1 - torch.eye(n).to(trans_logits.device)).bool()).view(n, n - 1)
        trans_exp_logits = trans_exp_logits.masked_select((1 - torch.eye(n).to(trans_logits.device)).bool()).view(n, n - 1)
       
        trans_diff = trans_diff.masked_select((1 - torch.eye(n).to(trans_logits.device)).bool()).view(n, n - 1)
        rot_diff = rot_diff.masked_select((1 - torch.eye(n).to(trans_logits.device)).bool()).view(n, n - 1)
        
        loss = 0.0
        
        rot_diff_expanded = rot_diff.unsqueeze(2).expand(-1, -1, n-1)
        rot_diff_expanded_columns = rot_diff_expanded.transpose(1, 2)
        
        trans_diff_expanded = trans_diff.unsqueeze(2).expand(-1, -1, n-1)
        trans_diff_expanded_columns = trans_diff_expanded.transpose(1, 2)
        
        rot_result_matrices = (rot_diff_expanded <= rot_diff_expanded_columns)
        trans_result_matrices = (trans_diff_expanded <= trans_diff_expanded_columns)
        
        hard_neg_mask = (rot_result_matrices * trans_result_matrices).float() # rot_diff<=anchor_rot_diff AND trans_diff<=anchor_trans_diff
        
        trans_neg_mask_label =  trans_result_matrices.permute(1,0,2)
        hard_neg_mask = hard_neg_mask.permute(1,0,2)
        
        # hard_neg_mask = (rot_result_matrices * trans_result_matrices).float() # rot_diff<=anchor_rot_diff AND trans_diff<=anchor_trans_diff
        # soft_neg_mask = (rot_result_matrices + trans_result_matrices).float() # rot_diff<=anchor_rot_diff OR trans_diff<=anchor_trans_diff
        
        
        # hard_neg_mask = hard_neg_mask.permute(1,0,2)
        # soft_neg_mask = soft_neg_mask.permute(1,0,2)
        
        eps = 1e-7
            
        pos_log_probs_trans = trans_logits.permute(1,0) - torch.log((trans_neg_mask_label * trans_exp_logits).sum(dim=-1)+eps)
        pos_log_probs_hard = trans_logits.permute(1,0) - torch.log((hard_neg_mask * trans_exp_logits).sum(dim=-1)+eps)
        
        loss_hard = - (pos_log_probs_hard / (n * (n - 1))).sum()
        loss_trans = - (pos_log_probs_trans / (n * (n - 1))).sum()
        
        loss = loss_hard + self.soft_lambda * loss_trans
        
        if not torch.isnan(loss) and not torch.isinf(loss): 
            return loss
        else:
            return None 
        
class RnCLoss_trans_mix(nn.Module):
    def __init__(self, temperature=2, soft_lambda=0.500, label_diff='l1', feature_sim='l2'):
        super(RnCLoss_trans_mix, self).__init__()
        self.t = temperature
        
        self.label_diff_fn_rot = LabelDifference_rot('cos')
        self.label_diff_fn = LabelDifference_trans(label_diff)
        self.feature_sim_fn = FeatureSimilarity(feature_sim)
        self.soft_lambda = soft_lambda

    def forward(self, trans_feat, rot, trans, complete=False):
        
        trans_diff = self.label_diff_fn(trans)
        
        rot_diff = self.label_diff_fn_rot(rot)
        
        trans_logits = self.feature_sim_fn(trans_feat).div(self.t)
        trans_logits_max, _ = torch.max(trans_logits, dim=1, keepdim=True)
        trans_logits -= trans_logits_max
        trans_exp_logits = trans_logits.exp()

        n = trans_logits.shape[0]  # n = 2bs

        # remove diagonal
        trans_logits = trans_logits.masked_select((1 - torch.eye(n).to(trans_logits.device)).bool()).view(n, n - 1)
        trans_exp_logits = trans_exp_logits.masked_select((1 - torch.eye(n).to(trans_logits.device)).bool()).view(n, n - 1)
       
        trans_diff = trans_diff.masked_select((1 - torch.eye(n).to(trans_logits.device)).bool()).view(n, n - 1)
        rot_diff = rot_diff.masked_select((1 - torch.eye(n).to(trans_logits.device)).bool()).view(n, n - 1)
        
        loss = 0.0
        
        rot_diff_expanded = rot_diff.unsqueeze(2).expand(-1, -1, n-1)
        rot_diff_expanded_columns = rot_diff_expanded.transpose(1, 2)
        
        trans_diff_expanded = trans_diff.unsqueeze(2).expand(-1, -1, n-1)
        trans_diff_expanded_columns = trans_diff_expanded.transpose(1, 2)
        
        rot_result_matrices = (rot_diff_expanded <= rot_diff_expanded_columns)
        trans_result_matrices = (trans_diff_expanded <= trans_diff_expanded_columns)
        
        hard_neg_mask = (rot_result_matrices * trans_result_matrices).float() # rot_diff<=anchor_rot_diff AND trans_diff<=anchor_trans_diff
        
        trans_neg_mask_label =  trans_result_matrices.permute(1,0,2)
        hard_neg_mask = hard_neg_mask.permute(1,0,2)
        
        # hard_neg_mask = (rot_result_matrices * trans_result_matrices).float() # rot_diff<=anchor_rot_diff AND trans_diff<=anchor_trans_diff
        # soft_neg_mask = (rot_result_matrices + trans_result_matrices).float() # rot_diff<=anchor_rot_diff OR trans_diff<=anchor_trans_diff
        
        
        # hard_neg_mask = hard_neg_mask.permute(1,0,2)
        # soft_neg_mask = soft_neg_mask.permute(1,0,2)
        
        eps = 1e-7
            
        pos_log_probs_trans = trans_logits.permute(1,0) - torch.log((trans_neg_mask_label * trans_exp_logits).sum(dim=-1)+eps)
        pos_log_probs_hard = trans_logits.permute(1,0) - torch.log((hard_neg_mask * trans_exp_logits).sum(dim=-1)+eps)
        
        loss_hard = - (pos_log_probs_hard / (n * (n - 1))).sum()
        loss_trans = - (pos_log_probs_trans / (n * (n - 1))).sum()
        
        loss = loss_hard + self.soft_lambda * loss_trans
        
        if not torch.isnan(loss) and not torch.isinf(loss): 
            return loss
        else:
            return None 
        



class RnCLoss_rot_mug_mix(nn.Module):
    def __init__(self, temperature=2, soft_lambda=0.900, label_diff='l1', feature_sim='l2'):
        super(RnCLoss_rot_mug_mix, self).__init__()
        self.t = temperature
        self.label_diff_fn = LabelDifference_rot(label_diff)
        self.label_diff_fn_trans = LabelDifference_trans('l1')
        
        self.feature_sim_fn = FeatureSimilarity(feature_sim)
        self.soft_lambda = soft_lambda

    def forward(self, rot_feat, rot_green, rot_red, trans, sym, complete=False):
        
        rot_green_diff = self.label_diff_fn(rot_green)
        rot_red_diff = self.label_diff_fn(rot_red)
        
        rot_diff = rot_green_diff + rot_red_diff
        trans_diff = self.label_diff_fn_trans(trans)
        
        sym_ind = (sym[:, 0] == 1).nonzero(as_tuple=True)[0] # all sym mug 
        rot_diff[sym_ind,:] = rot_green_diff[sym_ind,:] # for sym mug, only keep the green_vec difference between sym mug and other non-sym mug
        rot_diff[:,sym_ind] = rot_green_diff[:,sym_ind] # for sym mug, only keep the green_vec difference between sym mug and other non-sym mug
        
        rot_logits = self.feature_sim_fn(rot_feat).div(self.t)
        rot_logits_max, _ = torch.max(rot_logits, dim=1, keepdim=True)
        rot_logits -= rot_logits_max
        rot_exp_logits = rot_logits.exp()

        n = rot_logits.shape[0]  # n = 2bs

        # remove diagonal
        rot_logits = rot_logits.masked_select((1 - torch.eye(n).to(rot_logits.device)).bool()).view(n, n - 1)
        rot_exp_logits = rot_exp_logits.masked_select((1 - torch.eye(n).to(rot_logits.device)).bool()).view(n, n - 1)
       
        rot_diff = rot_diff.masked_select((1 - torch.eye(n).to(rot_logits.device)).bool()).view(n, n - 1)
        trans_diff = trans_diff.masked_select((1 - torch.eye(n).to(rot_logits.device)).bool()).view(n, n - 1)
        
        loss = 0.0
        
        rot_diff_expanded = rot_diff.unsqueeze(2).expand(-1, -1, n-1)
        rot_diff_expanded_columns = rot_diff_expanded.transpose(1, 2)
        trans_diff_expanded = trans_diff.unsqueeze(2).expand(-1, -1, n-1)
        trans_diff_expanded_columns = trans_diff_expanded.transpose(1, 2)

        rot_result_matrices = (rot_diff_expanded <= rot_diff_expanded_columns)
        trans_result_matrices = (trans_diff_expanded <= trans_diff_expanded_columns)
        
        hard_neg_mask = (rot_result_matrices * trans_result_matrices).float() # rot_diff<=anchor_rot_diff AND trans_diff<=anchor_trans_diff
        
        rot_neg_mask_label =  rot_result_matrices.permute(1,0,2)
        hard_neg_mask = hard_neg_mask.permute(1,0,2)
        
        # hard_neg_mask = (rot_result_matrices * trans_result_matrices).float() # rot_diff<=anchor_rot_diff AND trans_diff<=anchor_trans_diff
        # soft_neg_mask = (rot_result_matrices + trans_result_matrices).float() # rot_diff<=anchor_rot_diff OR trans_diff<=anchor_trans_diff
        
        
        # hard_neg_mask = hard_neg_mask.permute(1,0,2)
        # soft_neg_mask = soft_neg_mask.permute(1,0,2)
        
        eps = 1e-7
            
        pos_log_probs_rot = rot_logits.permute(1,0) - torch.log((rot_neg_mask_label * rot_exp_logits).sum(dim=-1)+eps)
        pos_log_probs_hard = rot_logits.permute(1,0) - torch.log((hard_neg_mask * rot_exp_logits).sum(dim=-1)+eps)
        
        loss_hard = - (pos_log_probs_hard / (n * (n - 1))).sum()
        loss_rot = - (pos_log_probs_rot / (n * (n - 1))).sum()
        
        loss = loss_hard + self.soft_lambda * loss_rot
        
        if not torch.isnan(loss) and not torch.isinf(loss): 
            return loss
        else:
            return None   


class RnCLoss_rot_nonSym_mix(nn.Module):
    def __init__(self, temperature=2, soft_lambda=0.900, label_diff='l1', feature_sim='l2'):
        super(RnCLoss_rot_nonSym_mix, self).__init__()
        self.t = temperature
        self.label_diff_fn = LabelDifference_rot(label_diff)
        self.label_diff_fn_trans = LabelDifference_trans('l1')
        self.feature_sim_fn = FeatureSimilarity(feature_sim)
        self.soft_lambda = soft_lambda

    def forward(self, rot_feat, rot_green, rot_red, trans, complete=False):
        
        rot_green_diff = self.label_diff_fn(rot_green)
        rot_red_diff = self.label_diff_fn(rot_red)
        
        rot_diff = rot_green_diff + rot_red_diff
        trans_diff = self.label_diff_fn_trans(trans)
        
        rot_logits = self.feature_sim_fn(rot_feat).div(self.t)
        rot_logits_max, _ = torch.max(rot_logits, dim=1, keepdim=True)
        rot_logits -= rot_logits_max
        rot_exp_logits = rot_logits.exp()

        n = rot_logits.shape[0]  # n = 2bs

        # remove diagonal
        rot_logits = rot_logits.masked_select((1 - torch.eye(n).to(rot_logits.device)).bool()).view(n, n - 1)
        rot_exp_logits = rot_exp_logits.masked_select((1 - torch.eye(n).to(rot_logits.device)).bool()).view(n, n - 1)
       
        rot_diff = rot_diff.masked_select((1 - torch.eye(n).to(rot_logits.device)).bool()).view(n, n - 1)
        trans_diff = trans_diff.masked_select((1 - torch.eye(n).to(rot_logits.device)).bool()).view(n, n - 1)
        
        loss = 0.0
        
        rot_diff_expanded = rot_diff.unsqueeze(2).expand(-1, -1, n-1)
        rot_diff_expanded_columns = rot_diff_expanded.transpose(1, 2)
        trans_diff_expanded = trans_diff.unsqueeze(2).expand(-1, -1, n-1)
        trans_diff_expanded_columns = trans_diff_expanded.transpose(1, 2)

        
        rot_result_matrices = (rot_diff_expanded <= rot_diff_expanded_columns)
        trans_result_matrices = (trans_diff_expanded <= trans_diff_expanded_columns)
        
        hard_neg_mask = (rot_result_matrices * trans_result_matrices).float() # rot_diff<=anchor_rot_diff AND trans_diff<=anchor_trans_diff
        
        rot_neg_mask_label =  rot_result_matrices.permute(1,0,2)
        hard_neg_mask = hard_neg_mask.permute(1,0,2)
        
        # hard_neg_mask = (rot_result_matrices * trans_result_matrices).float() # rot_diff<=anchor_rot_diff AND trans_diff<=anchor_trans_diff
        # soft_neg_mask = (rot_result_matrices + trans_result_matrices).float() # rot_diff<=anchor_rot_diff OR trans_diff<=anchor_trans_diff
        
        
        # hard_neg_mask = hard_neg_mask.permute(1,0,2)
        # soft_neg_mask = soft_neg_mask.permute(1,0,2)
        
        eps = 1e-7
            
        pos_log_probs_rot = rot_logits.permute(1,0) - torch.log((rot_neg_mask_label * rot_exp_logits).sum(dim=-1)+eps)
        pos_log_probs_hard = rot_logits.permute(1,0) - torch.log((hard_neg_mask * rot_exp_logits).sum(dim=-1)+eps)
        
        loss_hard = - (pos_log_probs_hard / (n * (n - 1))).sum()
        loss_rot = - (pos_log_probs_rot / (n * (n - 1))).sum()
        
        loss = loss_hard + self.soft_lambda * loss_rot
        
        if not torch.isnan(loss) and not torch.isinf(loss): 
            return loss
        else:
            return None      
class RnCLoss_rot_mix(nn.Module):
    def __init__(self, temperature=2, soft_lambda=0.900, label_diff='l1', feature_sim='l2'):
        super(RnCLoss_rot_mix, self).__init__()
        self.t = temperature
        self.label_diff_fn = LabelDifference_rot(label_diff)
        self.label_diff_fn_trans = LabelDifference_trans('l1')
        self.feature_sim_fn = FeatureSimilarity(feature_sim)
        self.soft_lambda = soft_lambda

    def forward(self, rot_feat, rot, trans, complete=False):
        
        rot_diff = self.label_diff_fn(rot)
        trans_diff = self.label_diff_fn_trans(trans)
        
        
        rot_logits = self.feature_sim_fn(rot_feat).div(self.t)
        rot_logits_max, _ = torch.max(rot_logits, dim=1, keepdim=True)
        rot_logits -= rot_logits_max
        rot_exp_logits = rot_logits.exp()

        
        n = rot_logits.shape[0]  # n = 2bs

        # remove diagonal
        rot_logits = rot_logits.masked_select((1 - torch.eye(n).to(rot_logits.device)).bool()).view(n, n - 1)
        rot_exp_logits = rot_exp_logits.masked_select((1 - torch.eye(n).to(rot_logits.device)).bool()).view(n, n - 1)
       
        rot_diff = rot_diff.masked_select((1 - torch.eye(n).to(rot_logits.device)).bool()).view(n, n - 1)
        trans_diff = trans_diff.masked_select((1 - torch.eye(n).to(rot_logits.device)).bool()).view(n, n - 1)
        
        loss = 0.0
        
        rot_diff_expanded = rot_diff.unsqueeze(2).expand(-1, -1, n-1)
        rot_diff_expanded_columns = rot_diff_expanded.transpose(1, 2)
        trans_diff_expanded = trans_diff.unsqueeze(2).expand(-1, -1, n-1)
        trans_diff_expanded_columns = trans_diff_expanded.transpose(1, 2)

        
        rot_result_matrices = (rot_diff_expanded <= rot_diff_expanded_columns)
        trans_result_matrices = (trans_diff_expanded <= trans_diff_expanded_columns)
        
        hard_neg_mask = (rot_result_matrices * trans_result_matrices).float() # rot_diff<=anchor_rot_diff AND trans_diff<=anchor_trans_diff
        
        rot_neg_mask_label =  rot_result_matrices.permute(1,0,2)
        hard_neg_mask = hard_neg_mask.permute(1,0,2)
        
        # hard_neg_mask = (rot_result_matrices * trans_result_matrices).float() # rot_diff<=anchor_rot_diff AND trans_diff<=anchor_trans_diff
        # soft_neg_mask = (rot_result_matrices + trans_result_matrices).float() # rot_diff<=anchor_rot_diff OR trans_diff<=anchor_trans_diff
        
        
        # hard_neg_mask = hard_neg_mask.permute(1,0,2)
        # soft_neg_mask = soft_neg_mask.permute(1,0,2)
        
        eps = 1e-7
            
        pos_log_probs_rot = rot_logits.permute(1,0) - torch.log((rot_neg_mask_label * rot_exp_logits).sum(dim=-1)+eps)
        pos_log_probs_hard = rot_logits.permute(1,0) - torch.log((hard_neg_mask * rot_exp_logits).sum(dim=-1)+eps)
        
        loss_hard = - (pos_log_probs_hard / (n * (n - 1))).sum()
        loss_rot = - (pos_log_probs_rot / (n * (n - 1))).sum()
        
        loss = loss_hard + self.soft_lambda * loss_rot

        if not torch.isnan(loss) and not torch.isinf(loss): 
            return loss
        else:
            return None  