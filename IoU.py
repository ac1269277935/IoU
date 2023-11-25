import torch

def Iou_loss(preds, bbox, eps=1e-6, reduction='mean'):
    '''
    preds:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    bbox:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    reduction:"mean"or"sum"
    return: loss
    '''
    # IOU是交并比，即A交B/A并B = A交B/A面积+B面积-A交B面积
    #
    x1 = torch.max(preds[:, 0], bbox[:, 0])
    y1 = torch.max(preds[:, 1], bbox[:, 1])
    x2 = torch.min(preds[:, 2], bbox[:, 2])
    y2 = torch.min(preds[:, 3], bbox[:, 3])

    # + 1.0 是为了防止边界框宽度为0的情况，如果两个坐标点完全相同，则加1确保宽度至少为1。
    # .clamp(0.) 这个方法的作用是限制结果的最小值为0。因为有时候计算出的宽度或高度可能是负数，而实际的边界框不可能有负的宽高，因此需要将其限制为0及以上。
    w = (x2 - x1 + 1.0).clamp(0.)
    h = (y2 - y1 + 1.0).clamp(0.)

    # 交集
    inters = w * h
    print("inters:\n", inters)

    # 并集
    uni = (preds[:, 2] - preds[:, 0] + 1.0) * (preds[:, 3] - preds[:, 1] + 1.0) + (bbox[:, 2] - bbox[:, 0] + 1.0) * (
            bbox[:, 3] - bbox[:, 1] + 1.0) - inters
    print("uni:\n", uni)

    # 交并比
    # .clamp(min=eps) 方法是将 IoU 值限制在一个小的正数 ε（一般取 ε=1e-5 或 ε=1e-6），
    # 主要是避免在交集区域很小或者没有交集的情况下出现除以零的错误。
    ious = (inters / uni).clamp(min=eps)

    # loss = -ious.log() 表示负对数损失函数
    # 我们的目标是在两个物体之间最大化它们的IoU值。因此，当我们试图最大化IoU时，实际上是试图最小化这个负对数损失函数。
    loss = -ious.log()

    if reduction == 'mean':
        loss = torch.mean(loss)
    elif reduction == 'sum':
        loss = torch.sum(loss)
    else:
        raise NotImplementedError
    print("last_loss:\n", loss)
    return loss


if __name__ == "__main__":
    # 输入预测值：二维tensor
    pred_box = torch.tensor([[2, 4, 6, 8], [5, 9, 13, 12]])

    # 输入真实值：二维tensor
    gt_box = torch.tensor([[3, 4, 7, 9]])
    loss = Iou_loss(preds=pred_box, bbox=gt_box)

# 输出结果
"""
inters:
 tensor([20.,  3.])
uni:
 tensor([35., 63.])
last_loss:
 tensor(1.8021)
"""
