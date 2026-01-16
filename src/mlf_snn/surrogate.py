"""
Surrogate Gradient 替代梯度函数
用于解决脉冲函数不可微的问题
"""

import torch
import torch.nn as nn
from typing import List


class SurrogateGradient(torch.autograd.Function):
    """基础替代梯度函数"""

    @staticmethod
    def forward(ctx, input, threshold=1.0):
        ctx.save_for_backward(input)
        ctx.threshold = threshold
        return (input >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # 使用sigmoid导数作为替代梯度
        alpha = 4.0
        grad = alpha * torch.sigmoid(alpha * (input - ctx.threshold)) * \
               (1 - torch.sigmoid(alpha * (input - ctx.threshold)))
        return grad_input * grad, None


class FastSigmoid(torch.autograd.Function):
    """
    Fast Sigmoid 替代梯度
    前向：阶跃函数
    反向：快速sigmoid近似
    """

    alpha = 4.0  # 控制梯度宽度

    @staticmethod
    def forward(ctx, input, threshold=1.0):
        ctx.save_for_backward(input)
        ctx.threshold = threshold
        return (input >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad = FastSigmoid.alpha / (2 * (1 + FastSigmoid.alpha *
               torch.abs(input - ctx.threshold)) ** 2)
        return grad_output * grad, None


class MultiThresholdSurrogate(torch.autograd.Function):
    """
    多阈值替代梯度函数
    支持 0.6/1.6/2.6 三阈值发放
    """

    alpha = 4.0

    @staticmethod
    def forward(ctx, u, thresholds=None):
        if thresholds is None:
            thresholds = [0.6, 1.6, 2.6]

        ctx.save_for_backward(u)
        ctx.thresholds = thresholds

        # 多阈值发放：输出 0, 1, 2, 3
        s = torch.zeros_like(u)
        for i, theta in enumerate(thresholds):
            s = torch.where(u >= theta, torch.tensor(i + 1.0, device=u.device), s)

        return s

    @staticmethod
    def backward(ctx, grad_output):
        u, = ctx.saved_tensors
        thresholds = ctx.thresholds
        alpha = MultiThresholdSurrogate.alpha

        # 多阈值梯度叠加
        grad = torch.zeros_like(u)
        for theta in thresholds:
            grad += alpha / (2 * (1 + alpha * torch.abs(u - theta)) ** 2)

        return grad_output * grad, None


class ATan(torch.autograd.Function):
    """Arctan 替代梯度"""

    alpha = 2.0

    @staticmethod
    def forward(ctx, input, threshold=1.0):
        ctx.save_for_backward(input)
        ctx.threshold = threshold
        return (input >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        alpha = ATan.alpha
        grad = alpha / (2 * (1 + (torch.pi / 2 * alpha *
               (input - ctx.threshold)) ** 2))
        return grad_output * grad, None


# 便捷函数
def spike_function(u, threshold=1.0, surrogate='fast_sigmoid'):
    """
    脉冲发放函数

    Args:
        u: 膜电位
        threshold: 阈值
        surrogate: 替代梯度类型
    """
    if surrogate == 'fast_sigmoid':
        return FastSigmoid.apply(u, threshold)
    elif surrogate == 'atan':
        return ATan.apply(u, threshold)
    else:
        return SurrogateGradient.apply(u, threshold)


def multi_spike_function(u, thresholds=None):
    """多阈值脉冲发放函数"""
    return MultiThresholdSurrogate.apply(u, thresholds)
