import torch

def Giou_loss(preds, bbox, eps=1e-7, reduction='mean'):
    '''
    https://github.com/sfzhang15/ATSS/blob/master/atss_core/modeling/rpn/atss/loss.py#L36
    :param preds:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    :param bbox:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    :return: loss
    '''
    ix1 = torch.max(preds[:, 0], bbox[:, 0])
    iy1 = torch.max(preds[:, 1], bbox[:, 1])
    ix2 = torch.min(preds[:, 2], bbox[:, 2])
    iy2 = torch.min(preds[:, 3], bbox[:, 3])

    iw = (ix2 - ix1 + 1.0).clamp(0.)
    ih = (iy2 - iy1 + 1.0).clamp(0.)

    # overlap
    # 交集
    inters = iw * ih
    print("inters:\n", inters)
    # union
    # 并集
    uni = (preds[:, 2] - preds[:, 0] + 1.0) * (preds[:, 3] - preds[:, 1] + 1.0) + (bbox[:, 2] - bbox[:, 0] + 1.0) * (
            bbox[:, 3] - bbox[:, 1] + 1.0) - inters + eps
    print("uni:\n", uni)
    # ious
    ious = inters / uni
    print("Iou:\n", ious)

    ex1 = torch.min(preds[:, 0], bbox[:, 0])
    ey1 = torch.min(preds[:, 1], bbox[:, 1])
    ex2 = torch.max(preds[:, 2], bbox[:, 2])
    ey2 = torch.max(preds[:, 3], bbox[:, 3])
    ew = (ex2 - ex1 + 1.0).clamp(min=0.)
    eh = (ey2 - ey1 + 1.0).clamp(min=0.)

    # enclose erea
    # C是最小封闭形状，正好可以把box A和B覆盖在内
    enclose = ew * eh + eps
    print("enclose:\n", enclose)

    giou = ious - (enclose - uni) / enclose
    loss = 1 - giou

    if reduction == 'mean':
        loss = torch.mean(loss)
    elif reduction == 'sum':
        loss = torch.sum(loss)
    else:
        raise NotImplementedError
    print("last_loss:\n", loss)
    return loss


if __name__ == "__main__":
    pred_box = torch.tensor([[2, 4, 6, 8], [5, 9, 13, 12]])
    gt_box = torch.tensor([[3, 4, 7, 9]])
    loss = Giou_loss(preds=pred_box, bbox=gt_box)

# 输出结果
"""
inters:
 tensor([20.,  3.])
uni:
 tensor([35., 63.])
Iou:
 tensor([0.5714, 0.0476])
enclose:
 tensor([36., 99.])
last_loss:
 tensor(0.8862)
"""
