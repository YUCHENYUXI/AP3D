import time
import torch
import torch.nn as nn
from tools.utils import AverageMeter
# TripletLoss 在 tools.losses 中定义
from tools.losses import TripletLoss

def train(epoch, model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu):
    batch_xent_loss = AverageMeter()
    batch_htri_loss = AverageMeter()
    batch_loss = AverageMeter()
    batch_corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    end = time.time()
    for batch_idx, (vids, pids, _) in enumerate(trainloader):
        if (pids - pids[0]).sum() == 0:
            # 当前batch中没有负样本，无法计算triplet loss
            continue

        if use_gpu:
            vids, pids = vids.cuda(), pids.cuda()

        data_time.update(time.time() - end)
        optimizer.zero_grad()

        # 前向传播
        outputs, features = model(vids)

        # 计算交叉熵损失和triplet损失
        xent_loss = criterion_xent(outputs, pids)
        htri_loss = criterion_htri(features, pids)
        loss = xent_loss + htri_loss

        # 反向传播和参数更新
        loss.backward()
        optimizer.step()

        # 统计正确率等指标
        _, preds = torch.max(outputs.data, 1)
        batch_corrects.update(torch.sum(preds == pids.data).float() / pids.size(0), pids.size(0))
        batch_xent_loss.update(xent_loss.item(), pids.size(0))
        batch_htri_loss.update(htri_loss.item(), pids.size(0))
        batch_loss.update(loss.item(), pids.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch{0} Time:{batch_time.sum:.1f}s Data:{data_time.sum:.1f}s Loss:{loss.avg:.4f} Xent:{xent.avg:.4f} Htri:{htri.avg:.4f} Acc:{acc.avg:.2%}'.format(
          epoch+1, batch_time=batch_time, data_time=data_time, loss=batch_loss,
          xent=batch_xent_loss, htri=batch_htri_loss, acc=batch_corrects))
