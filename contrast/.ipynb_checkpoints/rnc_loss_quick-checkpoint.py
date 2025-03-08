import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch3d

class LabelDifference(nn.Module):
    def __init__(self, distance_type='l1'):
        super(LabelDifference, self).__init__()
        self.distance_type = distance_type

    def forward(self, labels):
        # labels: [bs, label_dim]
        # output: [bs, bs]
        if self.distance_type == 'l1':
            return torch.abs(labels[:, None, :] - labels[None, :, :]).sum(dim=-1)
        elif self.distance_type == 'l2':
            return (labels[:, None, :] - labels[None, :, :]).norm(2, dim=-1) # posetive L2 norm
        else:
            raise ValueError(self.distance_type)


class FeatureSimilarity(nn.Module):
    def __init__(self, similarity_type='l2'):
        super(FeatureSimilarity, self).__init__()
        self.similarity_type = similarity_type

    def forward(self, features):
        # labels: [bs, feat_dim]
        # output: [bs, bs]
        if self.similarity_type == 'l2':
            
            return - (features[:, None, :] - features[None, :, :]).norm(2, dim=-1) # negative L2 norm
        else:
            raise ValueError(self.similarity_type)


class RnCLoss(nn.Module):
    def __init__(self, temperature=2, label_diff='l1', feature_sim='l2'):
        super(RnCLoss, self).__init__()
        self.t = temperature
        self.label_diff_fn = LabelDifference(label_diff)
        self.feature_sim_fn = FeatureSimilarity(feature_sim)

    def forward(self, features, labels):
        # features: [bs, 2, feat_dim]
        # labels: [bs, label_dim]
        
        # features = torch.cat([features[:, 0], features[:, 1]], dim=0)  # [2bs, feat_dim]
        # labels = labels.repeat(2, 1)  # [2bs, label_dim]

        label_diffs = self.label_diff_fn(labels)
        logits = self.feature_sim_fn(features).div(self.t)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits -= logits_max.detach()
        exp_logits = logits.exp()

        n = logits.shape[0]  # n = 2bs

        # remove diagonal
        logits = logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        exp_logits = exp_logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        label_diffs = label_diffs.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)

        loss = 0.
        for k in range(n - 1):
            pos_logits = logits[:, k]  # 2bs
            pos_label_diffs = label_diffs[:, k]  # 2bs
            neg_mask = (label_diffs >= pos_label_diffs.view(-1, 1)).float()  # [2bs, 2bs - 1]
            pos_log_probs = pos_logits - torch.log((neg_mask * exp_logits).sum(dim=-1))  # 2bs
            loss += - (pos_log_probs / (n * (n - 1))).sum()
        
        
        return loss

class RnCLoss_v2(nn.Module):
    def __init__(self, temperature=2, label_diff='l1', feature_sim='l2'):
        super(RnCLoss_v2, self).__init__()
        self.t = temperature
        self.label_diff_fn = LabelDifference(label_diff)
        self.feature_sim_fn = FeatureSimilarity(feature_sim)
        
    def pose_error(self, pose):
    
        rot = pytorch3d.transforms.rotation_6d_to_matrix(pose[:,:6])
        div = torch.pow((torch.linalg.det(rot)), 1/3)
        div = div.unsqueeze(dim=-1)
        div = div.unsqueeze(dim=-1)
        rot = rot / div
        t = pose[:, 6:]
        
        rot_trans = torch.transpose(rot, 1, 2)  # N*3*3
        rot = rot.unsqueeze(1) # N*1*3*3

        R = torch.matmul(rot, rot_trans)  # 2000*30000*3*3
        R_trace = torch.diagonal(R, offset=0, dim1=-1, dim2=-2).sum(-1)  # 2000*30000
        cos_theta = (R_trace - 1) / 2  # 2000*30000 , [0, 0] = -0.9055
        theta = torch.arccos(torch.clip(cos_theta, -1.0, 1.0)) * 180 / torch.pi
        
        t_reshaped = torch.tile(t.unsqueeze(1), (1, t.shape[0], 1))
        shift = torch.linalg.norm(t_reshaped - t, dim=-1) * 100.

        return theta, shift

    def forward(self, features, labels):

        theta_diffs, shift_diffs = self.pose_error(labels)
        logits = self.feature_sim_fn(features).div(self.t)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits -= logits_max.detach()
        exp_logits = logits.exp()

        n = logits.shape[0]  # n = 2bs

        # remove diagonal
        logits = logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        exp_logits = exp_logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        
        theta_diffs = theta_diffs.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        shift_diffs = shift_diffs.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)

        loss = 0.
        for k in range(n - 1):
            pos_logits = logits[:, k]  # 2bs
            pos_theta_diffs = theta_diffs[:, k]  # 2bs
            pos_shift_diffs = shift_diffs[:, k]  # 2bs
            neg_mask_theta = (theta_diffs >= pos_theta_diffs.view(-1, 1)).float()  # [2bs, 2bs - 1]
            neg_mask_shift = (shift_diffs >= pos_shift_diffs.view(-1, 1)).float()  # [2bs, 2bs - 1]
            pos_log_probs_theta = pos_logits - torch.log((neg_mask_theta * exp_logits).sum(dim=-1))  # 2bs
            pos_log_probs_shift = pos_logits - torch.log((neg_mask_shift * exp_logits).sum(dim=-1))  # 2bs
            
            pos_log_probs = (pos_log_probs_theta + pos_log_probs_shift) / 2.0
            loss += - (pos_log_probs / (n * (n - 1))).sum()
        
        
        return loss


class RnCLoss_v3(nn.Module):
    def __init__(self, temperature=2, label_diff='l1', feature_sim='l2'):
        super(RnCLoss_v3, self).__init__()
        self.t = temperature
        self.label_diff_fn = LabelDifference(label_diff)
        self.feature_sim_fn = FeatureSimilarity(feature_sim)
        
    def pose_error(self, pose):
    
        rot = pytorch3d.transforms.rotation_6d_to_matrix(pose[:,:6])
        div = torch.pow((torch.linalg.det(rot)), 1/3)
        div = div.unsqueeze(dim=-1)
        div = div.unsqueeze(dim=-1)
        rot = rot / div
        t = pose[:, 6:]
        
        rot_trans = torch.transpose(rot, 1, 2)  # N*3*3
        rot = rot.unsqueeze(1) # N*1*3*3

        R = torch.matmul(rot, rot_trans)  # 2000*30000*3*3
        R_trace = torch.diagonal(R, offset=0, dim1=-1, dim2=-2).sum(-1)  # 2000*30000
        cos_theta = (R_trace - 1) / 2  # 2000*30000 , [0, 0] = -0.9055
        theta = torch.arccos(torch.clip(cos_theta, -1.0, 1.0)) * 180 / torch.pi
        
        t_reshaped = torch.tile(t.unsqueeze(1), (1, t.shape[0], 1))
        shift = torch.linalg.norm(t_reshaped - t, dim=-1) * 100.

        return theta, shift

    def forward(self, theta_diffs, shift_diffs, features):
        
        logits = self.feature_sim_fn(features).div(self.t)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits -= logits_max.detach()
        exp_logits = logits.exp()

        n = logits.shape[0]  # n = 2bs

        # remove diagonal
        logits = logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        exp_logits = exp_logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        
        theta_diffs = theta_diffs.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        shift_diffs = shift_diffs.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)

        loss = 0.
        for k in range(n - 1):
            pos_logits = logits[:, k]  # 2bs
            pos_theta_diffs = theta_diffs[:, k]  # 2bs
            pos_shift_diffs = shift_diffs[:, k]  # 2bs
            neg_mask_theta = (theta_diffs >= pos_theta_diffs.view(-1, 1)).float()  # [2bs, 2bs - 1]
            neg_mask_shift = (shift_diffs >= pos_shift_diffs.view(-1, 1)).float()  # [2bs, 2bs - 1]
            pos_log_probs_theta = pos_logits - torch.log((neg_mask_theta * exp_logits).sum(dim=-1))  # 2bs
            pos_log_probs_shift = pos_logits - torch.log((neg_mask_shift * exp_logits).sum(dim=-1))  # 2bs
            
            pos_log_probs = (pos_log_probs_theta + pos_log_probs_shift) / 2.0
            loss += - (pos_log_probs / (n * (n - 1))).sum()
        
        
        return loss


class RnCLoss_v4(nn.Module):
    def __init__(self, temperature=2, label_diff='l1', feature_sim='l2'):
        super(RnCLoss_v4, self).__init__()
        self.t = temperature
        self.label_diff_fn = LabelDifference(label_diff)
        self.feature_sim_fn = FeatureSimilarity(feature_sim)
        
    def pose_error(self, pose):
    
        rot = pytorch3d.transforms.rotation_6d_to_matrix(pose[:,:6])
        div = torch.pow((torch.linalg.det(rot)), 1/3)
        div = div.unsqueeze(dim=-1)
        div = div.unsqueeze(dim=-1)
        rot = rot / div
        t = pose[:, 6:]
        
        rot_trans = torch.transpose(rot, 1, 2)  # N*3*3
        rot = rot.unsqueeze(1) # N*1*3*3

        R = torch.matmul(rot, rot_trans)  # 2000*30000*3*3
        R_trace = torch.diagonal(R, offset=0, dim1=-1, dim2=-2).sum(-1)  # 2000*30000
        cos_theta = (R_trace - 1) / 2  # 2000*30000 , [0, 0] = -0.9055
        theta = torch.arccos(torch.clip(cos_theta, -1.0, 1.0)) * 180 / torch.pi
        
        t_reshaped = torch.tile(t.unsqueeze(1), (1, t.shape[0], 1))
        shift = torch.linalg.norm(t_reshaped - t, dim=-1) * 100.

        return theta, shift

    def forward(self, theta_diffs, shift_diffs, features):

        
        logits = self.feature_sim_fn(features).div(self.t)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits -= logits_max.detach()
        exp_logits = logits.exp()

        n = logits.shape[0]  # n = 2bs

        # remove diagonal
        logits = logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        exp_logits = exp_logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        
        theta_diffs = theta_diffs.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        shift_diffs = shift_diffs.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        
        loss = 0.0
        # 将matrix1扩展到4x3x3，以便每一列都可以与整个矩阵进行比较
        theta_diffs_expanded = theta_diffs.unsqueeze(2).expand(-1, -1, 3)

        # 通过交换维度获取每列的副本，以便进行元素比较
        theta_diffs_expanded_columns = theta_diffs_expanded.transpose(1, 2)

        # 使用减法作为比较操作，执行元素比较
        result_matrices = (theta_diffs_expanded <= theta_diffs_expanded_columns).float()
        neg_mask_theta =  result_matrices.permute(1,0,2)

        # 将matrix1扩展到4x3x3，以便每一列都可以与整个矩阵进行比较
        shift_diffs_expanded = shift_diffs.unsqueeze(2).expand(-1, -1, 3)

        # 通过交换维度获取每列的副本，以便进行元素比较
        shift_diffs_expanded_columns = shift_diffs_expanded.transpose(1, 2)

        # 使用减法作为比较操作，执行元素比较
        result_matrices = (shift_diffs_expanded <= shift_diffs_expanded_columns).float()
        neg_mask_shift = result_matrices.permute(1,0,2)

        pos_log_probs_theta = logits.permute(1,0) - torch.log((neg_mask_theta * exp_logits).sum(dim=-1))
        pos_log_probs_shift = logits.permute(1,0) - torch.log((neg_mask_shift * exp_logits).sum(dim=-1))  # 2bs

        pos_log_probs = (pos_log_probs_theta + pos_log_probs_shift) / 2.0

        loss = - (pos_log_probs / (n * (n - 1))).sum()
        
        return loss

if __name__ == '__main__':
    theta_diffs, shift_diffs = torch.randn(8, 8), torch.randn(8, 8)
    features = torch.randn(8, 256)
    
    RnCLoss_v3 = RnCLoss_v3()
    loss = RnCLoss_v3(theta_diffs, shift_diffs,features)
    print(loss)
    

# +
# remove diagonal

bs = 128
logits = torch.randn(bs, bs)
exp_logits = torch.randn(bs, bs)

theta_diffs = torch.arange(12).reshape(4,3)
shift_diffs = torch.arange(12).reshape(4,3)

n = logits.shape[0]
# -

loss = 0.
for k in range(n - 1):
    
    pos_logits = logits[:, k]  # 2bs
    pos_theta_diffs = theta_diffs[:, k]  # 2bs
    pos_shift_diffs = shift_diffs[:, k]  # 2bs
    neg_mask_theta = (theta_diffs >= pos_theta_diffs.view(-1, 1)).float()  # [2bs, 2bs - 1]
    
    neg_mask_shift = (shift_diffs >= pos_shift_diffs.view(-1, 1)).float()  # [2bs, 2bs - 1]

    pos_log_probs_theta = pos_logits - torch.log((neg_mask_theta * exp_logits).sum(dim=-1))  # 2bs
    
    
    
    pos_log_probs_shift = pos_logits - torch.log((neg_mask_shift * exp_logits).sum(dim=-1))  # 2bs

    pos_log_probs = (pos_log_probs_theta + pos_log_probs_shift) / 2.0
    
    loss += - (pos_log_probs / (n * (n - 1))).sum()
print(loss)

# +
# 将matrix1扩展到4x3x3，以便每一列都可以与整个矩阵进行比较
theta_diffs_expanded = theta_diffs.unsqueeze(2).expand(-1, -1, 3)

# 通过交换维度获取每列的副本，以便进行元素比较
theta_diffs_expanded_columns = theta_diffs_expanded.transpose(1, 2)

# 使用减法作为比较操作，执行元素比较
result_matrices = (theta_diffs_expanded <= theta_diffs_expanded_columns).float()
neg_mask_theta =  result_matrices.permute(1,0,2)

# 将matrix1扩展到4x3x3，以便每一列都可以与整个矩阵进行比较
shift_diffs_expanded = shift_diffs.unsqueeze(2).expand(-1, -1, 3)

# 通过交换维度获取每列的副本，以便进行元素比较
shift_diffs_expanded_columns = shift_diffs_expanded.transpose(1, 2)

# 使用减法作为比较操作，执行元素比较
result_matrices = (shift_diffs_expanded <= shift_diffs_expanded_columns).float()
neg_mask_shift = result_matrices.permute(1,0,2)

pos_log_probs_theta = logits.permute(1,0) - torch.log((neg_mask_theta * exp_logits).sum(dim=-1))
pos_log_probs_shift = logits.permute(1,0) - torch.log((neg_mask_shift * exp_logits).sum(dim=-1))  # 2bs

pos_log_probs = (pos_log_probs_theta + pos_log_probs_shift) / 2.0

loss = - (pos_log_probs / (n * (n - 1))).sum()

print(loss)
# -


