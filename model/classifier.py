from torch import nn
from .modules import MishLayer


class Classifier(nn.Module):
	def __init__(self, inp_size: int, out_size: int, hid_size: int = 512):
		super(Classifier, self).__init__()

		self.model = nn.Sequential(
			#BatchSwapNoise(0.15),
			nn.Linear(inp_size, hid_size),
			nn.BatchNorm1d(hid_size),
                        MishLayer(),
			nn.Linear(hid_size, hid_size//2),
                        MishLayer(),
			nn.BatchNorm1d(hid_size//2),
			nn.Linear(hid_size//2, hid_size//4),
                        MishLayer(),
                        nn.BatchNorm1d(hid_size//4),
                        nn.Linear(hid_size//4, out_size)
			#nn.Softmax(-1),
		)

	def forward(self, z):
		validity = self.model(z)
		return validity
