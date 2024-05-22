import torch

class WeightedSumFunc(torch.autograd.Function):
    a_ij_require_grad=True

    @staticmethod
    def forward(ctx, a_ij, e_ijk):
        # a_ij (N, group_size) e_ijk (N, group_size, dim)
        ctx.save_for_backward(a_ij, e_ijk)
        output = torch.einsum('bi,bij->bj', a_ij, e_ijk)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # (N, dim)
        a_ij, e_ijk = ctx.saved_tensors
        if WeightedSumFunc.a_ij_require_grad:
            d_a_ij = torch.einsum('bj,bij->bi', grad_output.float(), e_ijk)
        else:
            d_a_ij = None
        d_e_ijk = torch.einsum('bi,bj->bij', a_ij, grad_output.float())
        return d_a_ij, d_e_ijk