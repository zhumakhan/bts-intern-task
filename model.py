import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(DoubleConv, self).__init__()

		self.conv = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),

			nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)

		)

	def forward(self, x):
		return self.conv(x)

class Unet(nn.Module):
	def __init__(self, in_channels=3, out_channels=1, features=[64,128,256,512]):
		super(Unet, self).__init__()
		
		self.downs = nn.ModuleList()
		self.ups = nn.ModuleList()

		self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

		# down
		for feature in features:
			self.downs.append(
				DoubleConv(in_channels, feature)
			)
			in_channels = feature

		# up

		for feature in reversed(features):
			self.ups.append(
				nn.ConvTranspose2d(
					feature*2, feature, kernel_size=2, stride=2
				)
			)

			self.ups.append(
				DoubleConv(feature*2, feature)
			)

		self.bottom_conv = DoubleConv(features[-1], features[-1]*2)
		self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

	def forward(self, x):
		skip_connections = []

		for down in self.downs:
			x = down(x)
			skip_connections.append(x)
			x = self.pool(x)

		x = self.bottom_conv(x)
		skip_connections = skip_connections[::-1]

		for idx in range(0, len(self.ups), 2):
			x = self.ups[idx](x)
			skip_connection = skip_connections[idx//2]

			if x.shape != skip_connection.shape:
				x = TF.resize(x, size=skip_connection.shape[2:])

			concat_skip = torch.cat((skip_connection, x), dim=1)
			x = self.ups[idx+1](concat_skip)

		return self.final_conv(x)


def test():
	x = torch.randn((3,1, 161, 201))
	model = Unet(in_channels=1, out_channels=1)
	y = model(x)

	print(x.shape)
	print(y.shape)

	assert x.shape == y.shape


if __name__ == '__main__':
	test()











