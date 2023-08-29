# from model.loss_metrics import supcon_loss
import torch
import torch.nn as nn
torch.manual_seed(1234)

def supcon_loss(input_feat, 
               labels = None, mask = None, sim_metric = None, 
               t=0.07, contra_mode='all', length_norm=False):
    """
    loss = SupConLoss(feat, 
                      labels = None, mask = None, sim_metric = None, 
                      t=0.07, contra_mode='all')
    input
    -----
      feat: tensor, feature vectors z [bsz, n_views, ...].
      labels: ground truth of shape [bsz].
      mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
            has the same class as sample i. Can be asymmetric.
      sim_metric: func, function to measure the similarity between two 
            feature vectors
      t: float, temperature
      contra_mode: str, default 'all'
         'all': use all data in class i as anchors
         'one': use 1st data in class i as anchors
      length_norm: bool, default False
          if True, l2 normalize feat along the last dimension

    output
    ------
      A loss scalar.
        
    Based on https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.

    Example:
      feature = torch.rand([16, 2, 1000], dtype=torch.float32)
      feature = torch_nn_func.normalize(feature, dim=-1)
      label = torch.tensor([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5, 1, 1, 1, 1, 1], 
               dtype=torch.long)
      loss = supcon_loss(feature, labels=label)
    """
    if length_norm:
        feat = torch.nn.normalize(input_feat, dim=-1)
    else:
        feat = input_feat
        
    # batch size
    bs = feat.shape[0]
    # device
    dc = feat.device
    # dtype
    dt = feat.dtype
    # number of view
    nv = feat.shape[1]
    
    # get the mask
    # mask[i][:] indicates the data that has the same class label as data i
    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = torch.eye(bs, dtype=dt, device=dc)
    elif labels is not None:
        labels = labels.view(-1, 1)
        if labels.shape[0] != bs:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).type(dt).to(dc)
    else:
        mask = mask.type(dt).to(dc)
    
    # prepare feature matrix
    # -> (num_view * batch, feature_dim, ...)
    contrast_feature = torch.cat(torch.unbind(feat, dim=1), dim=0)
    # 
    if contra_mode == 'one':
        # (batch, feat_dim, ...)
        anchor_feature = feat[:, 0]
        anchor_count = 1
    elif contra_mode == 'all':
        anchor_feature = contrast_feature
        anchor_count = nv
    else:
        raise ValueError('Unknown mode: {}'.format(contra_mode))
    
    # compute logits
    # logits_mat is a matrix of size [num_view * batch, num_view * batch]
    # or [batch, num_view * batch]
    
    print("anchor_feature.shape", anchor_feature.shape)
    print("contrast_feature.shape", contrast_feature.shape)

    if sim_metric is not None:
        logits_mat = torch.div(
            sim_metric(anchor_feature, contrast_feature), t)
    else:
        logits_mat = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), t)
    
    # print(anchor_feature.shape)
    # mask based on the label
    # -> same shape as logits_mat 
    mask_ = mask.repeat(anchor_count, nv)
    # mask on each data itself (
    self_mask = torch.scatter(
        torch.ones_like(mask_), 1, 
        torch.arange(bs * anchor_count).view(-1, 1).to(dc), 
        0)
    print(self_mask)
    # 
    mask_ = mask_ * self_mask
    print(mask_)
    
    # for numerical stability, remove the max from logits
    # see https://en.wikipedia.org/wiki/LogSumExp trick
    # for numerical stability
    # print("logits_mat * self_mask: \n",logits_mat * self_mask)
    logits_max, _ = torch.max(logits_mat * self_mask, dim=1, keepdim=True)
    logits_mat_ = logits_mat - logits_max.detach()
    # compute log_prob
    exp_logits = torch.exp(logits_mat_ * self_mask) * self_mask
    print("exp_logits\n", exp_logits)
    log_prob = logits_mat_ - torch.log(exp_logits.sum(1, keepdim=True))
    print("exp_logits.sum(1, keepdim=True)", exp_logits.sum(1, keepdim=True))

    print("log_prob\n", log_prob)
    # compute mean of log-likelihood over positive
    print(mask_ * log_prob)
    mean_log_prob_pos = (mask_ * log_prob).sum(1) / mask_.sum(1)
    print("mean_log_prob_pos.shape", mean_log_prob_pos.shape)
    print(mean_log_prob_pos)
    # loss
    loss = - mean_log_prob_pos
    print(loss.view(anchor_count, bs))
    loss = loss.view(anchor_count, bs).mean()

    return loss

def sim_metric_seq(mat1, mat2):
    if len(mat1.shape) == 2:
        mat1 = mat1.unsqueeze(-1)
        mat2 = mat2.unsqueeze(-1)
    return torch.bmm(mat1.permute(1, 0, 2), mat2.permute(1, 2, 0)).mean(0)

def loss_SCL(batch, K, S, t=0.07):
    """
    Computes the loss based on the given input batch, K, and S.

    Args:
    - batch (torch.Tensor): Input batch tensor of shape [1 + K + M + S, H ...]
    - K (int): Number of augmented samples
    - S (int): Number of negative samples

    Returns:
    - loss (torch.Tensor): Computed loss value
    """
    bsz = batch.shape[0]  # Get batch size
    device = batch.device  # Get device

    M = len(batch) - 1 - K - S  # Calculate the number of other real samples
    remove_rows = list(range(bsz))
    remove_rows[1:1+K+M] = [] # positive sample rows
    print("remove_rows\n", remove_rows)
    logits_mat = sim_metric_seq(batch, batch)
    # print("logits_mat\n", logits_mat)
    logits_mat = logits_mat[remove_rows]
    self_mask = torch.ones((bsz,bsz))
    self_mask.diagonal().fill_(0) # mask on each data itself
    self_mask[:,0].fill_(0) # no need to compute the loss of the anchor
    self_mask = self_mask[remove_rows]
    # print("self_mask\n",self_mask)
    self_mask = self_mask.to(device)
    
    logits_max, _ = torch.max(logits_mat * self_mask, dim=1, keepdim=True)
    logits_mat_ = logits_mat - logits_max.detach()
    # compute log_prob
    exp_logits = torch.exp(logits_mat_ * self_mask) * self_mask
    # divide by the sum of exp_logits along the row
    log_prob = logits_mat_ - torch.log(exp_logits.sum(1, keepdim=True))
    # print("log_prob\n", log_prob)
    
    mask_ = torch.ones((bsz,bsz))
    mask_.diagonal().fill_(0) 
    # mask_[1:1+K+M,:].fill_(0) # mask on augmented and positive samples
    mask_[0,1+K+M:].fill_(0) # mask on anchor to negative samples
    mask_[1+K+M:,1:1+K+M].fill_(0) # mask on negative samples to augmented and positive samples
    mask_ = mask_.to(device)
    
    # print("remove_rows", remove_rows)   
    
    mask_ = mask_[remove_rows]
    # log_prob = log_prob[remove_rows]
    # print("mask_\n", mask_)
    
    mean_log_prob_pos = (mask_ * log_prob).sum(1) / mask_.sum(1)
    # print("mean_log_prob_pos.shape", mean_log_prob_pos.shape)
    # print(mean_log_prob_pos)
    # loss
    loss = - mean_log_prob_pos
    loss = loss.mean()

    return loss

feature = torch.rand(8, 5)
# feature = torch.Tensor([[1,4,3,2,1],[1,4,3,2,1],[1,4,3,2,1],[1,4,3,2,1],[1,4,3,2,1],[1,4,3,2,1],[1,4,3,2,1],[1,4,3,2,1]])
# feature = feature.unsqueeze(1)
feature = feature.unsqueeze(-1)
print("feature.shape", feature.shape)
# print("feature", feature)
# feature = nn.functional.normalize(feature, dim=-1)
label = torch.tensor([1,1,1,1,0,0,0,0], 
               dtype=torch.long)
# loss = supcon_loss(feature, labels=label, contra_mode='all', sim_metric=sim_metric_seq)
loss = loss_SCL(feature, 1, 4)

print(loss)