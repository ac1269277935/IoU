import torch

def Diou_loss(preds, bbox, eps=1e-7, reduction='mean'):
    '''
    preds:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    bbox:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    eps: eps to avoid divide 0
    reduction: mean or sum
    return: diou-loss
    '''
    ix1 = torch.max(preds[:, 0], bbox[:, 0])
    iy1 = torch.max(preds[:, 1], bbox[:, 1])
    ix2 = torch.min(preds[:, 2], bbox[:, 2])
    iy2 = torch.min(preds[:, 3], bbox[:, 3])

    iw = (ix2 - ix1 + 1.0).clamp(min=0.)
    ih = (iy2 - iy1 + 1.0).clamp(min=0.)

    # overlaps
    # 交集
    inters = iw * ih

    # union
    uni = (preds[:, 2] - preds[:, 0] + 1.0) * (preds[:, 3] - preds[:, 1] + 1.0) + (bbox[:, 2] - bbox[:, 0] + 1.0) * (
            bbox[:, 3] - bbox[:, 1] + 1.0) - inters

    # iou
    iou = inters / (uni + eps)
    print("iou:\n", iou)

    # inter_diag
    cxpreds = (preds[:, 2] + preds[:, 0]) / 2
    cypreds = (preds[:, 3] + preds[:, 1]) / 2

    cxbbox = (bbox[:, 2] + bbox[:, 0]) / 2
    cybbox = (bbox[:, 3] + bbox[:, 1]) / 2

    # inter_diag是预测框和真实框两个中心点之间的欧氏距离
    inter_diag = (cxbbox - cxpreds) ** 2 + (cybbox - cypreds) ** 2
    print("inter_diag:\n", inter_diag)

    # outer_diag
    ox1 = torch.min(preds[:, 0], bbox[:, 0])
    oy1 = torch.min(preds[:, 1], bbox[:, 1])
    ox2 = torch.max(preds[:, 2], bbox[:, 2])
    oy2 = torch.max(preds[:, 3], bbox[:, 3])

    # outer_diag是正好包含预测框和真实框的最小框的对角线距离的平方
    outer_diag = (ox1 - ox2) ** 2 + (oy1 - oy2) ** 2
    print("outer_diag:\n", outer_diag)

    diou = iou - inter_diag / outer_diag
    diou = torch.clamp(diou, min=-1.0, max=1.0)

    diou_loss = 1 - diou
    print("last_loss:\n", diou_loss)

    if reduction == 'mean':
        loss = torch.mean(diou_loss)
    elif reduction == 'sum':
        loss = torch.sum(diou_loss)
    else:
        raise NotImplementedError
    return loss


if __name__ == "__main__":
    pred_box = torch.tensor([[2, 4, 6, 8], [5, 9, 13, 12]])
    gt_box = torch.tensor([[3, 4, 7, 9]])
    loss = Diou_loss(preds=pred_box, bbox=gt_box)

# 输出结果
"""
iou:
 tensor([0.5714, 0.0476])
inter_diag:
 tensor([ 1, 32])
outer_diag:
 tensor([ 50, 164])
last_loss:
 tensor([0.4286, 0.9524])
"""
