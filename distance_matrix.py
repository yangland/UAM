import torch
import torch.nn.functional as F
import math


class Distance():
    def __init__(self, device):
        self.device = device

    # L2 Distances Matrix
    @staticmethod
    def l2dist(p1, p2):
        """L2 distance between p1, p2, each of which is a tensor"""
        return torch.dist(p1, p2, p=2)

    # cos Distances Matrix
    @staticmethod
    def cosine_similarity(grad1, grad2, normalized=False):
        """
        Input: two sets of gradients of the same shape
        Output range: [-1, 1]
        """
        cos_sim = F.cosine_similarity(torch.flatten(grad1), torch.flatten(grad2), 0, 1e-15)
        if normalized:
            return (cos_sim + 1.0) / 2.0
        else:
            return cos_sim

    # Function for CKA
    def centering(self, K):
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device)
        I = torch.eye(n, device=self.device)
        H = I - unit / n
        return torch.matmul(torch.matmul(H, K), H)

    def rbf(self, X, sigma=None):
        GX = torch.matmul(X, X.T)
        KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
        if sigma is None:
            mdist = torch.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = torch.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return torch.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = torch.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = torch.sqrt(self.kernel_HSIC(Y, Y, sigma))
        return hsic / (var1 * var2)

