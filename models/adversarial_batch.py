import torch.autograd as autograd
from torch import nn


class GradientReversalF(autograd.Function):
    """
    Gradient Reversal Layer as a torch autograd function.
    """
    @staticmethod
    def forward(ctx, x, lambda_=1.0):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = -ctx.lambda_ * grad_output.clone()
        return grad_input, None


class GradientReversalL(nn.Module):
    def __init__(self, lambda_=1.0):
        super(GradientReversalL, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalF.apply(x, self.lambda_)


class AdversarialClassifier(nn.Module):
	def __init__(self, input_dim, output_dim):
		super(AdversarialClassifier, self).__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.grl = GradientReversalL(lambda_=1.0)
		self.model = nn.Linear(self.input_dim, self.output_dim)
		self.criterion = nn.CrossEntropyLoss()

	def forward(self, x):
		x_grl = self.grl(x)
		logits = self.model(x_grl)
		return logits

	def loss(self, logits, y):
		return self.criterion(logits, y)