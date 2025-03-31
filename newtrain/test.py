import time
import numpy as np
import torch
from tools.eval_metrics import evaluate

def test(model, queryloader, galleryloader, use_gpu, args):
    since = time.time()
    model.eval()

    qf, q_pids, q_camids = [], [], []
    # query ===
    for batch_idx, (vids, pids, camids) in enumerate(queryloader):
        if use_gpu:
            vids = vids.cuda()
        feat = model(vids)
        feat = feat.mean(1)
        feat = model.module.bn(feat)
        feat = feat.data.cpu()

        qf.append(feat)
        q_pids.extend(pids)
        q_camids.extend(camids)

    qf = torch.cat(qf, 0)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)
    print("Extracted features for query set, obtained {} matrix".format(qf.shape))
    # ===
    gf, g_pids, g_camids = [], [], []
    # gallery
    for batch_idx, (vids, pids, camids) in enumerate(galleryloader):
        if use_gpu:
            vids = vids.cuda()
        feat = model(vids)
        feat = feat.mean(1)
        feat = model.module.bn(feat)
        feat = feat.data.cpu()

        gf.append(feat)
        g_pids.extend(pids)
        g_camids.extend(camids)

    gf = torch.cat(gf, 0)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)
    
    if args.dataset == 'mars':
        # 对于 mars 数据集，gallery必须包含query set
        gf = torch.cat((qf, gf), 0)
        g_pids = np.append(q_pids, g_pids)
        g_camids = np.append(q_camids, g_camids)

    print("Extracted features for gallery set, obtained {} matrix".format(gf.shape))
    # ===

    time_elapsed = time.time() - since
    print('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print("Computing distance matrix")
    m, n = qf.size(0), gf.size(0)
    distmat = torch.zeros((m, n))

    if args.distance == 'euclidean':
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        for i in range(m):
            distmat[i:i+1].addmm_(1, -2, qf[i:i+1], gf.t())
    else:
        q_norm = torch.norm(qf, p=2, dim=1, keepdim=True)
        g_norm = torch.norm(gf, p=2, dim=1, keepdim=True)
        qf = qf.div(q_norm.expand_as(qf))
        gf = gf.div(g_norm.expand_as(gf))
        for i in range(m):
            distmat[i] = - torch.mm(qf[i:i+1], gf.t())

    distmat = distmat.numpy()

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    print("Results ----------")
    print('top1:{:.1%} top5:{:.1%} top10:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], mAP))
    print("------------------")

    return cmc[0]
