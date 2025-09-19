import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import re

def calculate_psnr_and_ssim(img1, img2):
    psnr_value = psnr(img1, img2)
    # ssim_value = ssim(img1, img2, multichannel=True)
    return psnr_value #, ssim_value

def batch_process_images(folder1, folder2):

    img_files_1 = sorted(os.listdir(folder1), key=lambda x: int(re.search(r'(\d+)', x).group(0)) if re.search(r'(\d+)', x) else x)
    img_files_2 = sorted(os.listdir(folder2), key=lambda x: int(re.search(r'(\d+)', x).group(0)) if re.search(r'(\d+)', x) else x)

    psnr_values = []
    # ssim_values = []

    for img_file_1, img_file_2 in zip(img_files_1, img_files_2):
        if not img_file_1.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff')):
            continue
        if not img_file_2.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff')):
            continue

        img_path_1 = os.path.join(folder1, img_file_1)
        img_path_2 = os.path.join(folder2, img_file_2)

        img1 = cv2.imread(img_path_1)
        img2 = cv2.imread(img_path_2)

        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # psnr_value, ssim_value = calculate_psnr_and_ssim(img1_rgb, img2_rgb)
        psnr_value = calculate_psnr_and_ssim(img1_rgb, img2_rgb)

        psnr_values.append(psnr_value)
        # ssim_values.append(ssim_value)

        print(f"Image pair: {img_file_1} vs {img_file_2}")
        print(f"PSNR: {psnr_value:.4f}")

    avg_psnr = np.mean(psnr_values)
    # avg_ssim = np.mean(ssim_values)

    print("\nAverage PSNR:", avg_psnr)
    # print("Average SSIM:", avg_ssim)

if __name__ == "__main__":
    folder1 = ''
    folder2 = ''

    batch_process_images(folder1, folder2)


'''
ValueError: win_size exceeds image extent. Either ensure that your images are at least 7x7; or pass win_size explicitly in the function call, with an odd value less than or equal to the smaller side of your images.
If your images are multichannel (with color channels), set channel_axis to the axis number corresponding to the channels.
'''
# --------------------------------ssim---------------------------
import torch
import numpy as np

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        # print(img1.size())
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)



if __name__ == '__main__':

    import os
    import os
    from PIL import Image
    import torch
    import torchvision.transforms as transforms
    from torch.autograd import Variable
    import torch.nn.functional as F
    from math import exp
    import re

    def sorted_files(file_list):
        return sorted(file_list, key=lambda x: int(re.search(r'(\d+)', x).group(0)) if re.search(r'(\d+)', x) else x)


    def compute_ssim_between_folders(folder1, folder2, window_size=11, size_average=True):

        image_files1 = [f for f in os.listdir(folder1) if f.endswith(('png', 'jpg', 'jpeg'))]
        image_files2 = [f for f in os.listdir(folder2) if f.endswith(('png', 'jpg', 'jpeg'))]

        image_files1 = sorted_files(image_files1)
        image_files2 = sorted_files(image_files2)

        assert len(image_files1) == len(image_files2), "两个文件夹中的图片数量不一致！"

        ssim_calculator = SSIM(window_size=window_size, size_average=size_average)

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        ssim_values = []

        for i in range(len(image_files1)):
            img1_path = os.path.join(folder1, image_files1[i])
            img2_path = os.path.join(folder2, image_files2[i])

            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')

            img1_tensor = transform(img1).unsqueeze(0)
            img2_tensor = transform(img2).unsqueeze(0)

            ssim_value = ssim_calculator(img1_tensor, img2_tensor)
            ssim_values.append(ssim_value.item())

            print(f"Image pair: {image_files1[i]} vs {image_files2[i]}")
            print(f"PSNR: {ssim_value:.4f}")

        average_ssim = sum(ssim_values) / len(ssim_values) if ssim_values else 0
        return average_ssim


    folder1 = ''
    folder2 = ''
    average_ssim = compute_ssim_between_folders(folder1, folder2)

    print(f"两个文件夹中所有对应图片的平均SSIM值: {average_ssim}")
