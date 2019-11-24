import torch
from torchvision import transforms
from dataset import NYUDataset
from custom_transforms import *
import plot_utils
import model_utils
from nn_model import Net
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#%matplotlib inline
import torch.nn.functional as F
import torch.nn as nn
import math

bs = 8
sz = (320,240)
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
mean, std = torch.tensor(mean), torch.tensor(std)
unnormalize = UnNormalizeImgBatch(mean, std)

tfms = transforms.Compose([
    ResizeImgAndDepth(sz),
    RandomHorizontalFlip(),
    ImgAndDepthToTensor(),
    NormalizeImg(mean, std)
])

ds = NYUDataset('/content/gdrive/My Drive/data/', tfms) # 경록
dl = torch.utils.data.DataLoader(ds, bs, shuffle=True)

model = Net()
model.to(device)

model.load_state_dict(torch.load('/content/gdrive/My Drive/data/answer.ckpt', map_location="cpu")) # 경록
criterion = nn.MSELoss()


model.eval()
error_0 = 0 # scale-Invariant Error
error_1 = 0 # RMS linear
error_2 = 0 # RMS log
error_3 = 0 # abs rel
error_4 = 0 # sqr rel
avg_psnr = 0 # ㅔpsnr
with torch.no_grad():
    data, target = next(iter(dl))
    data, target = data.to(device), target.to(device)
    output = model(data)
    print(f"{len(data)}")
    error_0 += model_utils.depth_loss(output, target).item()
    error_1 += model_utils.err_rms_linear(output, target).item()
    error_2 += model_utils.err_rms_log(output, target).item()
    error_3 += model_utils.err_abs_rel(output, target).item()
    error_4 += model_utils.err_sql_rel(output, target).item()

    #psnr을 위해서 가공 중.
    output = (output * 0.225) + 0.45
    output = output * 255
    output[output <= 0] = 0.00001
    target[target == 0] = 0.00001
    target.unsqueeze_(dim=1) # actual_depth 를
    print(f"{output.shape}")
    print(f"{target.shape}")
    mse = criterion(output, target)
    psnr = 10 * math.log10(120*160 / mse.item())
    avg_psnr += psnr

    error_0 /= len(data)
    error_1 /= len(data)
    error_2 /= len(data)
    error_3 /= len(data)
    error_4 /= len(data)
    avg_psnr /= len(data)
    print('test is over')
    print(f'Test set: Average loss:{error_0:.4f} / {error_1:.4f} / {error_2:.4f} /{error_3:.4f} /{error_4:.4f}  /{avg_psnr:.4f}')
