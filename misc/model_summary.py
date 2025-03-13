from torchsummary import summary

from src.models import Generator

resNet = Generator(
    ngf=64, n_downsampling=3, norm_type="batch", g_type="resnet-9", device="cuda"
)

if __name__ == "__main__":
    summary(resNet, (3, 512, 768))
