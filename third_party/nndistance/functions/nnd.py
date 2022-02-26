# functions/add.py
import torch
from torch.autograd import Function
#from _ext import my_lib
import my_lib_cuda as my_lib


class NNDFunction(Function):

    @staticmethod
    def forward(ctx, xyz1, xyz2):
        device = xyz1.device
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()   

        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)
        
        idx1 = torch.zeros(batchsize, n).type(torch.IntTensor)
        idx2 = torch.zeros(batchsize, m).type(torch.IntTensor)
        
        if not xyz1.is_cuda:
            my_lib.nnd_forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        else:
            dist1 = dist1.cuda()
            dist2 = dist2.cuda()
            idx1 = idx1.cuda()
            idx2 = idx2.cuda()
            my_lib.nnd_forward_cuda(xyz1, xyz2, dist1, dist2, idx1, idx2)

        ctx.save_for_backward(xyz1,xyz2,dist1,dist2,idx1,idx2)
        idx1 = idx1.to(device=device, dtype=torch.long)
        idx2 = idx2.to(device=device, dtype=torch.long)
        return dist1, dist2, idx1, idx2

    @staticmethod
    def backward(ctx, graddist1, graddist2, gradidx1, gradidx2):

        xyz1, xyz2, dist1, dist2, idx1, idx2 = ctx.saved_tensors
        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()

        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())
        
        if not graddist1.is_cuda:
            my_lib.nnd_backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
        else:
            gradxyz1 = gradxyz1.cuda()
            gradxyz2 = gradxyz2.cuda()
            my_lib.nnd_backward_cuda(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
        return gradxyz1, gradxyz2