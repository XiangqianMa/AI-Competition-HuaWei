import numpy as np
import torch


def generate_mixed_sample(beta, sample, target):
    """生成cutmix样本和类标

    Args:
        beta: beta参数
        sample: 原始样本
        target: 原始类标
    Returns:
        sample: 转换后的样本
        target_a: 类标a
        target_b: 类标b
    """
    # generate mixed sample
    lam = np.random.beta(beta, beta)
    rand_index = torch.randperm(sample.size()[0]).cuda()
    target_a = target
    target_b = target[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(sample.size(), lam)
    sample[:, :, bbx1:bbx2, bby1:bby2] = sample[rand_index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (sample.size()[-1] * sample.size()[-2]))
    
    return sample, target_a, target_b, lam


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2