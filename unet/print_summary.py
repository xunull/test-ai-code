from torchinfo import summary
from unet import UNet

model = UNet(3, 1)
batch_size = 1
summary(model, input_size=(batch_size, 3, 572, 572))
