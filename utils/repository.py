import numpy as np
import torch

class TSRepository(object):
    def __init__(self, n, dim, num_classes, temperature):
        self.n = n
        self.dim = dim 
        self.features = torch.FloatTensor(self.n, self.dim)
        self.targets = torch.LongTensor(self.n)
        self.ptr = 0
        self.device = 'cpu'
        self.K = 100
        self.temperature = temperature
        self.C = num_classes

    def weighted_knn(self, predictions):
        # perform weighted knn
        retrieval_one_hot = torch.zeros(self.K, self.C).to(self.device)
        batchSize = predictions.shape[0]
        correlation = torch.matmul(predictions, self.features.t())
        yd, yi = correlation.topk(self.K, dim=1, largest=True, sorted=True)
        candidates = self.targets.view(1,-1).expand(batchSize, -1)
        retrieval = torch.gather(candidates, 1, yi)
        retrieval_one_hot.resize_(batchSize * self.K, self.C).zero_()
        retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
        yd_transform = yd.clone().div_(self.temperature).exp_()
        probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , self.C), 
                          yd_transform.view(batchSize, -1, 1)), 1)
        _, class_preds = probs.sort(1, True)
        class_pred = class_preds[:, 0]

        return class_pred

    def knn(self, predictions):
        # perform knn
        correlation = torch.matmul(predictions, self.features.t())
        sample_pred = torch.argmax(correlation, dim=1)
        class_pred = torch.index_select(self.targets, 0, sample_pred)
        return class_pred

    def mine_nearest_neighbors(self, topk, calculate_accuracy=True):
        far, near = self.furthest_nearest_neighbors(topk)
        if calculate_accuracy:
            targets = self.targets[:self.ptr].cpu().numpy()
            accuracy = np.mean(targets[near] == targets[:, None])
            return far, near, accuracy
        return far, near

    def furthest_nearest_neighbors(self, topk):
        """Exact per-sample neighbors; exclude self, including tied vectors.

        Block distances limit temporary memory; computation remains quadratic.
        Only populated repository entries participate.
        """
        if topk < 1 or self.ptr < 2:
            raise ValueError('Neighbor mining needs topk >= 1 and at least two samples.')
        features = self.features[:self.ptr].detach().cpu()
        if not torch.isfinite(features).all():
            raise ValueError('Cannot mine neighbors from non-finite features.')
        n = len(features)
        k = min(topk, n - 1)
        nearest = np.empty((n, k), dtype=np.int64)
        furthest = np.empty((n, k), dtype=np.int64)
        block_size = max(1, min(256, 4_000_000 // n))
        for start in range(0, n, block_size):
            end = min(n, start + block_size)
            distances = torch.cdist(features[start:end], features)
            rows = torch.arange(end - start)
            cols = torch.arange(start, end)
            distances[rows, cols] = float('inf')
            nearest[start:end] = distances.topk(k, largest=False).indices.numpy()
            distances[rows, cols] = -float('inf')
            furthest[start:end] = distances.topk(k, largest=True).indices.numpy()
        return furthest, nearest


    def reset(self):
        self.ptr = 0

    def resize(self, sz):
        self.n = sz * self.n
        self.features = torch.FloatTensor(self.n, self.dim)
        self.targets = torch.LongTensor(self.n)
        
    def update(self, features, targets):
        b = features.size(0)
        
        assert(b + self.ptr <= self.n)
        
        self.features[self.ptr:self.ptr+b].copy_(features.detach())
        if not torch.is_tensor(targets): targets = torch.from_numpy(targets)
        self.targets[self.ptr:self.ptr+b].copy_(targets.detach())
        self.ptr += b

    def to(self, device):
        self.features = self.features.to(device)
        self.targets = self.targets.to(device)
        self.device = device

    def cpu(self):
        self.to('cpu')

    def cuda(self):
        self.to('cuda:0')
