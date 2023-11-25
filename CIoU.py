import math
import torch

def Ciou_loss(preds, bbox, eps=1e-7, reduction='mean'):
    '''
    https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/loss/multibox_loss.py
    :param preds:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    :param bbox:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    :param eps: eps to avoid divide 0
    :param reduction: mean or sum
    :return: diou-loss
    '''
    ix1 = torch.max(preds[:, 0], bbox[:, 0])
    iy1 = torch.max(preds[:, 1], bbox[:, 1])
    ix2 = torch.min(preds[:, 2], bbox[:, 2])
    iy2 = torch.min(preds[:, 3], bbox[:, 3])

    iw = (ix2 - ix1 + 1.0).clamp(min=0.)
    ih = (iy2 - iy1 + 1.0).clamp(min=0.)

    # overlaps
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

    inter_diag = (cxbbox - cxpreds) ** 2 + (cybbox - cypreds) ** 2

    # outer_diag
    ox1 = torch.min(preds[:, 0], bbox[:, 0])
    oy1 = torch.min(preds[:, 1], bbox[:, 1])
    ox2 = torch.max(preds[:, 2], bbox[:, 2])
    oy2 = torch.max(preds[:, 3], bbox[:, 3])

    outer_diag = (ox1 - ox2) ** 2 + (oy1 - oy2) ** 2

    diou = iou - inter_diag / outer_diag
    print("diou:\n", diou)

    # calculate v,alpha
    wbbox = bbox[:, 2] - bbox[:, 0] + 1.0
    hbbox = bbox[:, 3] - bbox[:, 1] + 1.0
    wpreds = preds[:, 2] - preds[:, 0] + 1.0
    hpreds = preds[:, 3] - preds[:, 1] + 1.0
    v = torch.pow((torch.atan(wbbox / hbbox) - torch.atan(wpreds / hpreds)), 2) * (4 / (math.pi ** 2))
    alpha = v / (1 - iou + v)
    ciou = diou - alpha * v
    ciou = torch.clamp(ciou, min=-1.0, max=1.0)

    ciou_loss = 1 - ciou
    if reduction == 'mean':
        loss = torch.mean(ciou_loss)
    elif reduction == 'sum':
        loss = torch.sum(ciou_loss)
    else:
        raise NotImplementedError
    print("last_loss:\n", loss)
    return loss


if __name__ == "__main__":
    pred_box = torch.tensor([[2, 4, 6, 8], [5, 9, 13, 12]])
    gt_box = torch.tensor([[3, 4, 7, 9]])
    loss = Ciou_loss(preds=pred_box, bbox=gt_box)

# 输出结果
"""
iou:
 tensor([0.5714, 0.0476])
diou:
 tensor([0.5714, 0.0476])
last_loss:
 tensor(0.6940)
"""
