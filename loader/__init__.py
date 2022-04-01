import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from PIL import Image
from config import cfg


from loader.stanford import get_stanford

def get_loader():
    pair = {

        'stanford': get_stanford,

    }

    return pair[cfg.data.type]()
