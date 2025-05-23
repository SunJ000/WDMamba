
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

import torch
import torchvision.utils as tvu
from torchvision import models
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.utils import get_root_logger, imwrite, tensor2img, img2tensor
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
import copy

import pyiqa
from .cal_ssim import SSIM
from torch import nn
import sys

def Normalize(x):
    ymax = 255
    ymin = 0
    xmax = x.max()
    xmin = x.min()
    return (ymax-ymin)*(x-xmin)/(xmax-xmin) + ymin


def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 0)


# 使用哈尔 haar 小波变换来实现二维离散小波
def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = int(in_batch/(r**2)),in_channel, r * in_height, r * in_width
    x1 = x[0:out_batch, :, :] / 2
    x2 = x[out_batch:out_batch * 2, :, :, :] / 2
    x3 = x[out_batch * 2:out_batch * 3, :, :, :] / 2
    x4 = x[out_batch * 3:out_batch * 4, :, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height,
                     out_width]).float().to(x.device)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  # 信号处理，非卷积运算，不需要进行梯度求导

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):  # 逆变换
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).cuda().features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):  # 1, 3, 256, 256
        h_relu1 = self.slice1(X)  # 1, 64, 256, 256
        h_relu2 = self.slice2(h_relu1)  # 1, 128, 128, 128
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)  # 1, 512, 16, 16
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]

class ContrastLoss(nn.Module):
    def __init__(self, ablation=False):

        super(ContrastLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.ab = ablation

    def forward(self, a, p, n):
        a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)
        loss = 0

        d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):
            d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
            if not self.ab:  # 根据ablation参数决定是否计算锚点样本与负样本之间的特征差异d_an
                d_an = self.l1(a_vgg[i], n_vgg[i].detach())
                contrastive = d_ap / (d_an + 1e-7)
            else:
                contrastive = d_ap

            loss += self.weights[i] * contrastive  # 不同层级特征加权损失
        return loss


@MODEL_REGISTRY.register()
class FeMaSRModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)

        # 定义网络
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.ssim = SSIM().cuda()
        self.l1 = nn.L1Loss().cuda()
        self.LL = None
        self.x_stage1 = None
        self.cr = ContrastLoss()

        # 敌营评价指标函数
        if self.opt['val'].get('metrics') is not None:
            self.metric_funcs = {}
            for _, opt in self.opt['val']['metrics'].items():
                mopt = opt.copy()
                name = mopt.pop('type', None)
                mopt.pop('better', None)
                self.metric_funcs[name] = pyiqa.create_metric(name, device=self.device, **mopt)

        # 加载预先训练的HQ ckpt、冻结解码器和码本
        self.LQ_stage = self.opt['network_g'].get('LQ_stage', False)
        if self.LQ_stage:
            load_path = self.opt['path'].get('pretrain_network_hq', None)
            assert load_path is not None, 'Need to specify hq prior model path in LQ stage'

            # hq_opt = self.opt['network_g'].copy()
            # hq_opt['LQ_stage'] = False
            # self.net_hq = build_network(hq_opt)
            # self.net_hq = self.model_to_device(self.net_hq)
            # self.load_network(self.net_hq, load_path, self.opt['path']['strict_load'])

            self.load_network(self.net_g, load_path, False)
            # frozen_module_keywords = self.opt['network_g'].get('frozen_module_keywords', None)
            # if frozen_module_keywords is not None:
            #     for name, module in self.net_g.named_modules():
            #         for fkw in frozen_module_keywords:
            #             if fkw in name:
            #                 for p in module.parameters():
            #                     p.requires_grad = False
            #                 break

        # 加载预训练模型
        load_path = self.opt['path'].get('pretrain_network_g', None)
        # print('#########################################################################',load_path)
        logger = get_root_logger()
        if load_path is not None:
            logger.info(f'Loading net_g from {load_path}')
            self.load_network(self.net_g, load_path, self.opt['path']['strict_load'])

        if self.is_train:
            self.init_training_settings()
            # self.use_dis = (self.opt['train']['gan_opt']['loss_weight'] != 0)
            # self.net_d_best = copy.deepcopy(self.net_d)

        self.net_g_best = copy.deepcopy(self.net_g)

    def init_training_settings(self):
        logger = get_root_logger()
        train_opt = self.opt['train']
        self.net_g.train()

        # define network net_d
        # self.net_d = build_network(self.opt['network_d'])
        # self.net_d = self.model_to_device(self.net_d)
        # load pretrained d models
        # load_path = self.opt['path'].get('pretrain_network_d', None)
        # # print(load_path)
        # if load_path is not None:
        #     logger.info(f'Loading net_d from {load_path}')
        #     self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True))

        # self.net_d.train()

        # 定义损失函数
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('fft_opt'):
            self.cri_fft = build_loss(train_opt['fft_opt']).to(self.device)
        else:
            self.cri_fft = None

        # if train_opt.get('perceptual_opt'):
        #     self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        #     self.model_to_device(self.cri_perceptual)
        # else:
        #     self.cri_perceptual = None

        # if train_opt.get('gan_opt'):
        #     self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        # self.net_d_iters = train_opt.get('net_d_iters', 1)
        # self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            optim_params.append(v)
            if not v.requires_grad:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        # 定义优化器
        optim_type = train_opt['optim_g'].pop('type')
        optim_class = getattr(torch.optim, optim_type)
        self.optimizer_g = optim_class(optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

        # optimizer d
        # optim_type = train_opt['optim_d'].pop('type')
        # optim_class = getattr(torch.optim, optim_type)
        # self.optimizer_d = optim_class(self.net_d.parameters(), **train_opt['optim_d'])
        # self.optimizers.append(self.optimizer_d)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        # self.lq_equalize = data['lq_equalize'].to(self.device)

        if 'gt' in data:
            self.gt = data['gt'].to(self.device)


    def print_network(self, model):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print("The number of parameters: {}".format(num_params))

    def optimize_parameters(self, current_iter):
        train_opt = self.opt['train']

        # for p in self.net_d.parameters():
        #     p.requires_grad = False
        self.optimizer_g.zero_grad()

        self.LL, self.x_stage1, self.output = self.net_g(self.lq)  # LL, x_stage1 和 x_final
        # self.LL, self.output = self.net_g(self.lq)  # LL, x_stage1

        # self.output = self.net_g(self.lq)  # DEN

        # if current_iter==0:

        l_g_total = 0
        loss_dict = OrderedDict()

        dwt = DWT()
        n, c, h, w = self.gt.shape

        gt_dwt = dwt(self.gt)
        gt_LL, gt_high0 = gt_dwt[:n, ...], gt_dwt[n:, ...]

        l_pix = self.l1(self.output, self.gt) + (1 - self.ssim(self.output, self.gt)) * 0.1
        l_g_total += l_pix
        loss_dict['l_pix'] = l_pix

        l_LL = self.l1(self.LL, gt_LL)
        l_g_total += l_LL
        loss_dict['l_LL'] = l_LL

        l_cr = 0.75 * self.cr(self.output, self.gt, self.lq) + 1.25 * self.cr(self.output, self.gt, self.x_stage1)

        l_g_total += 0.001 * l_cr
        loss_dict['l_cr'] = l_cr

        if train_opt.get('fft_opt', None):
            l_fft = self.cri_fft(self.output, self.gt)
            l_g_total += 0.1 * l_fft
            loss_dict['l_freq'] = l_fft

        l_g_total.mean().backward()

        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_g.eval()
        net_g = self.get_bare_model(self.net_g)
        min_size = 8000 * 8000  # use smaller min_size with limited GPU memory
        lq_input = self.lq
        # restoration = self.net_g(self.lq)
        _, _, h, w = lq_input.shape
        if h * w < min_size:
            # out_img, feature_degradation, self.output = self.net_g(self.lq, feature=feature_degradation)
            self.output = net_g.test(lq_input)
        else:
            self.output = net_g.test_tile(lq_input)
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, epoch, tb_logger, save_img, save_as_dir=None):
        logger = get_root_logger()
        logger.info('Only support single GPU validation.')
        self.nondist_validation(dataloader, current_iter, epoch, tb_logger, save_img, save_as_dir)

    def nondist_validation(self, dataloader, current_iter, epoch, tb_logger,
                           save_img, save_as_dir):
        # dataset_name = dataloader.dataset.opt['name']
        dataset_name = 'NTIRE2024'
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }

        pbar = tqdm(total=len(dataloader), unit='image')

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)

            # zero self.metric_results
            self.metric_results = {metric: 0 for metric in self.metric_results}
            self.key_metric = self.opt['val'].get('key_metric')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            sr_img = tensor2img(self.output)
            metric_data = [img2tensor(sr_img).unsqueeze(0) / 255, self.gt]

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], 'image_results',
                                             f'{current_iter}',
                                             f'{img_name}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["name"]}.png')
                if save_as_dir:
                    save_as_img_path = osp.join(save_as_dir, f'{img_name}.png')
                    imwrite(sr_img, save_as_img_path)
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    tmp_result = self.metric_funcs[name](*metric_data)
                    self.metric_results[name] += tmp_result.item()

            pbar.update(1)
            pbar.set_description(f'Test {img_name}')

        pbar.close()

        if with_metrics:
            # calculate average metric
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            if self.key_metric is not None:
                # If the best metric is updated, update and save best model
                to_update = self._update_best_metric_result(dataset_name, self.key_metric,
                                                            self.metric_results[self.key_metric], current_iter)

                if to_update:
                    for name, opt_ in self.opt['val']['metrics'].items():
                        self._update_metric_result(dataset_name, name, self.metric_results[name], current_iter)
                    self.copy_model(self.net_g, self.net_g_best)
                    # self.copy_model(self.net_d, self.net_d_best)
                    self.save_network(self.net_g, 'net_g_best', current_iter, epoch)
                    # self.save_network(self.net_d, 'net_d_best', current_iter, epoch)
            else:
                # update each metric separately
                updated = []
                for name, opt_ in self.opt['val']['metrics'].items():
                    tmp_updated = self._update_best_metric_result(dataset_name, name, self.metric_results[name],
                                                                  current_iter)
                    updated.append(tmp_updated)
                # save best model if any metric is updated
                if sum(updated):
                    self.copy_model(self.net_g, self.net_g_best)
                    # self.copy_model(self.net_d, self.net_d_best)
                    self.save_network(self.net_g, 'net_g_best', '')
                    # self.save_network(self.net_d, 'net_d_best', '')

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
#        if tb_logger:
#            for metric, value in self.metric_results.items():
#                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def vis_single_code(self, up_factor=2):
        net_g = self.get_bare_model(self.net_g)
        codenum = self.opt['network_g']['codebook_params'][0][1]
        with torch.no_grad():
            code_idx = torch.arange(codenum).reshape(codenum, 1, 1, 1)
            code_idx = code_idx.repeat(1, 1, up_factor, up_factor)
            output_img = net_g.decode_indices(code_idx)
            output_img = tvu.make_grid(output_img, nrow=32)

        return output_img.unsqueeze(0)

    def get_current_visuals(self):
        vis_samples = 16
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()[:vis_samples]
        out_dict['result'] = self.output.detach().cpu()[:vis_samples]
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()[:vis_samples]
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter, epoch)
        # self.save_network(self.net_d, 'net_d', current_iter, epoch)
        self.save_training_state(epoch, current_iter)









#
# # wo_cr
# from collections import OrderedDict
# from os import path as osp
# from tqdm import tqdm
#
# import torch
# import torchvision.utils as tvu
# from torchvision import models
# from basicsr.archs import build_network
# from basicsr.losses import build_loss
# from basicsr.utils import get_root_logger, imwrite, tensor2img, img2tensor
# from basicsr.utils.registry import MODEL_REGISTRY
# from .base_model import BaseModel
# import copy
#
# import pyiqa
# from .cal_ssim import SSIM
# from torch import nn
# import sys
#
# def Normalize(x):
#     ymax = 255
#     ymin = 0
#     xmax = x.max()
#     xmin = x.min()
#     return (ymax-ymin)*(x-xmin)/(xmax-xmin) + ymin
#
#
# def dwt_init(x):
#
#     x01 = x[:, :, 0::2, :] / 2
#     x02 = x[:, :, 1::2, :] / 2
#     x1 = x01[:, :, :, 0::2]
#     x2 = x02[:, :, :, 0::2]
#     x3 = x01[:, :, :, 1::2]
#     x4 = x02[:, :, :, 1::2]
#     x_LL = x1 + x2 + x3 + x4
#     x_HL = -x1 - x2 + x3 + x4
#     x_LH = -x1 + x2 - x3 + x4
#     x_HH = x1 - x2 - x3 + x4
#
#     return torch.cat((x_LL, x_HL, x_LH, x_HH), 0)
#
#
# # 使用哈尔 haar 小波变换来实现二维离散小波
# def iwt_init(x):
#     r = 2
#     in_batch, in_channel, in_height, in_width = x.size()
#     out_batch, out_channel, out_height, out_width = int(in_batch/(r**2)),in_channel, r * in_height, r * in_width
#     x1 = x[0:out_batch, :, :] / 2
#     x2 = x[out_batch:out_batch * 2, :, :, :] / 2
#     x3 = x[out_batch * 2:out_batch * 3, :, :, :] / 2
#     x4 = x[out_batch * 3:out_batch * 4, :, :, :] / 2
#
#     h = torch.zeros([out_batch, out_channel, out_height,
#                      out_width]).float().to(x.device)
#
#     h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
#     h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
#     h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
#     h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
#
#     return h
#
#
# class DWT(nn.Module):
#     def __init__(self):
#         super(DWT, self).__init__()
#         self.requires_grad = False  # 信号处理，非卷积运算，不需要进行梯度求导
#
#     def forward(self, x):
#         return dwt_init(x)
#
#
# class IWT(nn.Module):  # 逆变换
#     def __init__(self):
#         super(IWT, self).__init__()
#         self.requires_grad = False
#
#     def forward(self, x):
#         return iwt_init(x)
#
#
# @MODEL_REGISTRY.register()
# class FeMaSRModel(BaseModel):
#     def __init__(self, opt):
#         super().__init__(opt)
#
#         # 定义网络
#         self.net_g = build_network(opt['network_g'])
#         self.net_g = self.model_to_device(self.net_g)
#         self.ssim = SSIM().cuda()
#         self.l1 = nn.L1Loss().cuda()
#         self.LL = None
#
#         # 敌营评价指标函数
#         if self.opt['val'].get('metrics') is not None:
#             self.metric_funcs = {}
#             for _, opt in self.opt['val']['metrics'].items():
#                 mopt = opt.copy()
#                 name = mopt.pop('type', None)
#                 mopt.pop('better', None)
#                 self.metric_funcs[name] = pyiqa.create_metric(name, device=self.device, **mopt)
#
#         # 加载预先训练的HQ ckpt、冻结解码器和码本
#         self.LQ_stage = self.opt['network_g'].get('LQ_stage', False)
#         if self.LQ_stage:
#             load_path = self.opt['path'].get('pretrain_network_hq', None)
#             assert load_path is not None, 'Need to specify hq prior model path in LQ stage'
#
#             # hq_opt = self.opt['network_g'].copy()
#             # hq_opt['LQ_stage'] = False
#             # self.net_hq = build_network(hq_opt)
#             # self.net_hq = self.model_to_device(self.net_hq)
#             # self.load_network(self.net_hq, load_path, self.opt['path']['strict_load'])
#
#             self.load_network(self.net_g, load_path, False)
#             # frozen_module_keywords = self.opt['network_g'].get('frozen_module_keywords', None)
#             # if frozen_module_keywords is not None:
#             #     for name, module in self.net_g.named_modules():
#             #         for fkw in frozen_module_keywords:
#             #             if fkw in name:
#             #                 for p in module.parameters():
#             #                     p.requires_grad = False
#             #                 break
#
#         # 加载预训练模型
#         load_path = self.opt['path'].get('pretrain_network_g', None)
#         # print('#########################################################################',load_path)
#         logger = get_root_logger()
#         if load_path is not None:
#             logger.info(f'Loading net_g from {load_path}')
#             self.load_network(self.net_g, load_path, self.opt['path']['strict_load'])
#
#         if self.is_train:
#             self.init_training_settings()
#             # self.use_dis = (self.opt['train']['gan_opt']['loss_weight'] != 0)
#             # self.net_d_best = copy.deepcopy(self.net_d)
#
#         self.net_g_best = copy.deepcopy(self.net_g)
#
#     def init_training_settings(self):
#         logger = get_root_logger()
#         train_opt = self.opt['train']
#         self.net_g.train()
#
#         # define network net_d
#         # self.net_d = build_network(self.opt['network_d'])
#         # self.net_d = self.model_to_device(self.net_d)
#         # load pretrained d models
#         # load_path = self.opt['path'].get('pretrain_network_d', None)
#         # # print(load_path)
#         # if load_path is not None:
#         #     logger.info(f'Loading net_d from {load_path}')
#         #     self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True))
#
#         # self.net_d.train()
#
#         # 定义损失函数
#         if train_opt.get('pixel_opt'):
#             self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
#         else:
#             self.cri_pix = None
#
#         if train_opt.get('fft_opt'):
#             self.cri_fft = build_loss(train_opt['fft_opt']).to(self.device)
#         else:
#             self.cri_fft = None
#
#         # if train_opt.get('perceptual_opt'):
#         #     self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
#         #     self.model_to_device(self.cri_perceptual)
#         # else:
#         #     self.cri_perceptual = None
#
#         # if train_opt.get('gan_opt'):
#         #     self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)
#
#         # self.net_d_iters = train_opt.get('net_d_iters', 1)
#         # self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)
#
#         # set up optimizers and schedulers
#         self.setup_optimizers()
#         self.setup_schedulers()
#
#     def setup_optimizers(self):
#         train_opt = self.opt['train']
#         optim_params = []
#         for k, v in self.net_g.named_parameters():
#             optim_params.append(v)
#             if not v.requires_grad:
#                 logger = get_root_logger()
#                 logger.warning(f'Params {k} will not be optimized.')
#
#         # 定义优化器
#         optim_type = train_opt['optim_g'].pop('type')
#         optim_class = getattr(torch.optim, optim_type)
#         self.optimizer_g = optim_class(optim_params, **train_opt['optim_g'])
#         self.optimizers.append(self.optimizer_g)
#
#         # optimizer d
#         # optim_type = train_opt['optim_d'].pop('type')
#         # optim_class = getattr(torch.optim, optim_type)
#         # self.optimizer_d = optim_class(self.net_d.parameters(), **train_opt['optim_d'])
#         # self.optimizers.append(self.optimizer_d)
#
#     def feed_data(self, data):
#         self.lq = data['lq'].to(self.device)
#         # self.lq_equalize = data['lq_equalize'].to(self.device)
#
#         if 'gt' in data:
#             self.gt = data['gt'].to(self.device)
#
#
#     def print_network(self, model):
#         num_params = 0
#         for p in model.parameters():
#             num_params += p.numel()
#         print(model)
#         print("The number of parameters: {}".format(num_params))
#
#     def optimize_parameters(self, current_iter):
#         train_opt = self.opt['train']
#
#         # for p in self.net_d.parameters():
#         #     p.requires_grad = False
#         self.optimizer_g.zero_grad()
#
#         self.LL, self.output = self.net_g(self.lq)  # LL和 x_final
#
#         # if current_iter==0:
#
#         l_g_total = 0
#         loss_dict = OrderedDict()
#
#         dwt = DWT()
#         n, c, h, w = self.gt.shape
#
#         gt_dwt = dwt(self.gt)
#         gt_LL, gt_high0 = gt_dwt[:n, ...], gt_dwt[n:, ...]
#
#
#         l_pix = self.l1(self.output, self.gt) + (1 - self.ssim(self.output, self.gt)) * 0.1
#         l_g_total += l_pix
#         loss_dict['l_pix'] = l_pix
#
#         l_LL = self.l1(self.LL, gt_LL)
#         l_g_total += l_LL
#         loss_dict['l_LL'] = l_LL
#
#         if train_opt.get('fft_opt', None):
#             l_fft = self.cri_fft(self.output, self.gt)
#             l_g_total += 0.1 * l_fft
#             loss_dict['l_freq'] = l_fft
#
#         l_g_total.mean().backward()
#
#         self.optimizer_g.step()
#
#         self.log_dict = self.reduce_loss_dict(loss_dict)
#
#     def test(self):
#         self.net_g.eval()
#         net_g = self.get_bare_model(self.net_g)
#         min_size = 8000 * 8000  # use smaller min_size with limited GPU memory
#         lq_input = self.lq
#         # restoration = self.net_g(self.lq)
#         _, _, h, w = lq_input.shape
#         if h * w < min_size:
#             # out_img, feature_degradation, self.output = self.net_g(self.lq, feature=feature_degradation)
#             self.output = net_g.test(lq_input)
#         else:
#             self.output = net_g.test_tile(lq_input)
#         self.net_g.train()
#
#     def dist_validation(self, dataloader, current_iter, epoch, tb_logger, save_img, save_as_dir=None):
#         logger = get_root_logger()
#         logger.info('Only support single GPU validation.')
#         self.nondist_validation(dataloader, current_iter, epoch, tb_logger, save_img, save_as_dir)
#
#     def nondist_validation(self, dataloader, current_iter, epoch, tb_logger,
#                            save_img, save_as_dir):
#         # dataset_name = dataloader.dataset.opt['name']
#         dataset_name = 'NTIRE2024'
#         with_metrics = self.opt['val'].get('metrics') is not None
#         if with_metrics:
#             self.metric_results = {
#                 metric: 0
#                 for metric in self.opt['val']['metrics'].keys()
#             }
#
#         pbar = tqdm(total=len(dataloader), unit='image')
#
#         if with_metrics:
#             if not hasattr(self, 'metric_results'):  # only execute in the first run
#                 self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
#             # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
#             self._initialize_best_metric_results(dataset_name)
#
#             # zero self.metric_results
#             self.metric_results = {metric: 0 for metric in self.metric_results}
#             self.key_metric = self.opt['val'].get('key_metric')
#
#         for idx, val_data in enumerate(dataloader):
#             img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
#             self.feed_data(val_data)
#             self.test()
#
#             sr_img = tensor2img(self.output)
#             metric_data = [img2tensor(sr_img).unsqueeze(0) / 255, self.gt]
#
#             # tentative for out of GPU memory
#             del self.lq
#             del self.output
#             torch.cuda.empty_cache()
#
#             if save_img:
#                 if self.opt['is_train']:
#                     save_img_path = osp.join(self.opt['path']['visualization'], 'image_results',
#                                              f'{current_iter}',
#                                              f'{img_name}.png')
#                 else:
#                     if self.opt['val']['suffix']:
#                         save_img_path = osp.join(
#                             self.opt['path']['visualization'], dataset_name,
#                             f'{img_name}_{self.opt["val"]["suffix"]}.png')
#                     else:
#                         save_img_path = osp.join(
#                             self.opt['path']['visualization'], dataset_name,
#                             f'{img_name}_{self.opt["name"]}.png')
#                 if save_as_dir:
#                     save_as_img_path = osp.join(save_as_dir, f'{img_name}.png')
#                     imwrite(sr_img, save_as_img_path)
#                 imwrite(sr_img, save_img_path)
#
#             if with_metrics:
#                 # calculate metrics
#                 for name, opt_ in self.opt['val']['metrics'].items():
#                     tmp_result = self.metric_funcs[name](*metric_data)
#                     self.metric_results[name] += tmp_result.item()
#
#             pbar.update(1)
#             pbar.set_description(f'Test {img_name}')
#
#         pbar.close()
#
#         if with_metrics:
#             # calculate average metric
#             for metric in self.metric_results.keys():
#                 self.metric_results[metric] /= (idx + 1)
#
#             if self.key_metric is not None:
#                 # If the best metric is updated, update and save best model
#                 to_update = self._update_best_metric_result(dataset_name, self.key_metric,
#                                                             self.metric_results[self.key_metric], current_iter)
#
#                 if to_update:
#                     for name, opt_ in self.opt['val']['metrics'].items():
#                         self._update_metric_result(dataset_name, name, self.metric_results[name], current_iter)
#                     self.copy_model(self.net_g, self.net_g_best)
#                     # self.copy_model(self.net_d, self.net_d_best)
#                     self.save_network(self.net_g, 'net_g_best', current_iter, epoch)
#                     # self.save_network(self.net_d, 'net_d_best', current_iter, epoch)
#             else:
#                 # update each metric separately
#                 updated = []
#                 for name, opt_ in self.opt['val']['metrics'].items():
#                     tmp_updated = self._update_best_metric_result(dataset_name, name, self.metric_results[name],
#                                                                   current_iter)
#                     updated.append(tmp_updated)
#                 # save best model if any metric is updated
#                 if sum(updated):
#                     self.copy_model(self.net_g, self.net_g_best)
#                     # self.copy_model(self.net_d, self.net_d_best)
#                     self.save_network(self.net_g, 'net_g_best', '')
#                     # self.save_network(self.net_d, 'net_d_best', '')
#
#             self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
#
#     def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
#         log_str = f'Validation {dataset_name}\n'
#         for metric, value in self.metric_results.items():
#             log_str += f'\t # {metric}: {value:.4f}'
#             if hasattr(self, 'best_metric_results'):
#                 log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
#                             f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
#             log_str += '\n'
#
#         logger = get_root_logger()
#         logger.info(log_str)
# #        if tb_logger:
# #            for metric, value in self.metric_results.items():
# #                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)
#
#     def vis_single_code(self, up_factor=2):
#         net_g = self.get_bare_model(self.net_g)
#         codenum = self.opt['network_g']['codebook_params'][0][1]
#         with torch.no_grad():
#             code_idx = torch.arange(codenum).reshape(codenum, 1, 1, 1)
#             code_idx = code_idx.repeat(1, 1, up_factor, up_factor)
#             output_img = net_g.decode_indices(code_idx)
#             output_img = tvu.make_grid(output_img, nrow=32)
#
#         return output_img.unsqueeze(0)
#
#     def get_current_visuals(self):
#         vis_samples = 16
#         out_dict = OrderedDict()
#         out_dict['lq'] = self.lq.detach().cpu()[:vis_samples]
#         out_dict['result'] = self.output.detach().cpu()[:vis_samples]
#         if hasattr(self, 'gt'):
#             out_dict['gt'] = self.gt.detach().cpu()[:vis_samples]
#         return out_dict
#
#     def save(self, epoch, current_iter):
#         self.save_network(self.net_g, 'net_g', current_iter, epoch)
#         # self.save_network(self.net_d, 'net_d', current_iter, epoch)
#         self.save_training_state(epoch, current_iter)
#









# std cr
# from collections import OrderedDict
# from os import path as osp
# from tqdm import tqdm
#
# import torch
# import torchvision.utils as tvu
# from torchvision import models
# from basicsr.archs import build_network
# from basicsr.losses import build_loss
# from basicsr.utils import get_root_logger, imwrite, tensor2img, img2tensor
# from basicsr.utils.registry import MODEL_REGISTRY
# from .base_model import BaseModel
# import copy
#
# import pyiqa
# from .cal_ssim import SSIM
# from torch import nn
# import sys
#
# def Normalize(x):
#     ymax = 255
#     ymin = 0
#     xmax = x.max()
#     xmin = x.min()
#     return (ymax-ymin)*(x-xmin)/(xmax-xmin) + ymin
#
#
# def dwt_init(x):
#
#     x01 = x[:, :, 0::2, :] / 2
#     x02 = x[:, :, 1::2, :] / 2
#     x1 = x01[:, :, :, 0::2]
#     x2 = x02[:, :, :, 0::2]
#     x3 = x01[:, :, :, 1::2]
#     x4 = x02[:, :, :, 1::2]
#     x_LL = x1 + x2 + x3 + x4
#     x_HL = -x1 - x2 + x3 + x4
#     x_LH = -x1 + x2 - x3 + x4
#     x_HH = x1 - x2 - x3 + x4
#
#     return torch.cat((x_LL, x_HL, x_LH, x_HH), 0)
#
#
# # 使用哈尔 haar 小波变换来实现二维离散小波
# def iwt_init(x):
#     r = 2
#     in_batch, in_channel, in_height, in_width = x.size()
#     out_batch, out_channel, out_height, out_width = int(in_batch/(r**2)),in_channel, r * in_height, r * in_width
#     x1 = x[0:out_batch, :, :] / 2
#     x2 = x[out_batch:out_batch * 2, :, :, :] / 2
#     x3 = x[out_batch * 2:out_batch * 3, :, :, :] / 2
#     x4 = x[out_batch * 3:out_batch * 4, :, :, :] / 2
#
#     h = torch.zeros([out_batch, out_channel, out_height,
#                      out_width]).float().to(x.device)
#
#     h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
#     h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
#     h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
#     h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
#
#     return h
#
#
# class DWT(nn.Module):
#     def __init__(self):
#         super(DWT, self).__init__()
#         self.requires_grad = False  # 信号处理，非卷积运算，不需要进行梯度求导
#
#     def forward(self, x):
#         return dwt_init(x)
#
#
# class IWT(nn.Module):  # 逆变换
#     def __init__(self):
#         super(IWT, self).__init__()
#         self.requires_grad = False
#
#     def forward(self, x):
#         return iwt_init(x)
#
#
# class Vgg19(torch.nn.Module):
#     def __init__(self, requires_grad=False):
#         super(Vgg19, self).__init__()
#         vgg_pretrained_features = models.vgg19(pretrained=True).cuda().features
#         self.slice1 = torch.nn.Sequential()
#         self.slice2 = torch.nn.Sequential()
#         self.slice3 = torch.nn.Sequential()
#         self.slice4 = torch.nn.Sequential()
#         self.slice5 = torch.nn.Sequential()
#         for x in range(2):
#             self.slice1.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(2, 7):
#             self.slice2.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(7, 12):
#             self.slice3.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(12, 21):
#             self.slice4.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(21, 30):
#             self.slice5.add_module(str(x), vgg_pretrained_features[x])
#         if not requires_grad:
#             for param in self.parameters():
#                 param.requires_grad = False
#
#     def forward(self, X):  # 1, 3, 256, 256
#         h_relu1 = self.slice1(X)  # 1, 64, 256, 256
#         h_relu2 = self.slice2(h_relu1)  # 1, 128, 128, 128
#         h_relu3 = self.slice3(h_relu2)
#         h_relu4 = self.slice4(h_relu3)
#         h_relu5 = self.slice5(h_relu4)  # 1, 512, 16, 16
#         return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
#
# class ContrastLoss(nn.Module):
#     def __init__(self, ablation=False):
#
#         super(ContrastLoss, self).__init__()
#         self.vgg = Vgg19().cuda()
#         self.l1 = nn.L1Loss()
#         self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
#         self.ab = ablation
#
#     def forward(self, a, p, n):
#         a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)
#         loss = 0
#
#         d_ap, d_an = 0, 0
#         for i in range(len(a_vgg)):
#             d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
#             if not self.ab:  # 根据ablation参数决定是否计算锚点样本与负样本之间的特征差异d_an
#                 d_an = self.l1(a_vgg[i], n_vgg[i].detach())
#                 contrastive = d_ap / (d_an + 1e-7)
#             else:
#                 contrastive = d_ap
#
#             loss += self.weights[i] * contrastive  # 不同层级特征加权损失
#         return loss
#
#
# @MODEL_REGISTRY.register()
# class FeMaSRModel(BaseModel):
#     def __init__(self, opt):
#         super().__init__(opt)
#
#         # 定义网络
#         self.net_g = build_network(opt['network_g'])
#         self.net_g = self.model_to_device(self.net_g)
#         self.ssim = SSIM().cuda()
#         self.l1 = nn.L1Loss().cuda()
#         self.LL = None
#         self.cr = ContrastLoss()
#
#         # 敌营评价指标函数
#         if self.opt['val'].get('metrics') is not None:
#             self.metric_funcs = {}
#             for _, opt in self.opt['val']['metrics'].items():
#                 mopt = opt.copy()
#                 name = mopt.pop('type', None)
#                 mopt.pop('better', None)
#                 self.metric_funcs[name] = pyiqa.create_metric(name, device=self.device, **mopt)
#
#         # 加载预先训练的HQ ckpt、冻结解码器和码本
#         self.LQ_stage = self.opt['network_g'].get('LQ_stage', False)
#         if self.LQ_stage:
#             load_path = self.opt['path'].get('pretrain_network_hq', None)
#             assert load_path is not None, 'Need to specify hq prior model path in LQ stage'
#
#             # hq_opt = self.opt['network_g'].copy()
#             # hq_opt['LQ_stage'] = False
#             # self.net_hq = build_network(hq_opt)
#             # self.net_hq = self.model_to_device(self.net_hq)
#             # self.load_network(self.net_hq, load_path, self.opt['path']['strict_load'])
#
#             self.load_network(self.net_g, load_path, False)
#             # frozen_module_keywords = self.opt['network_g'].get('frozen_module_keywords', None)
#             # if frozen_module_keywords is not None:
#             #     for name, module in self.net_g.named_modules():
#             #         for fkw in frozen_module_keywords:
#             #             if fkw in name:
#             #                 for p in module.parameters():
#             #                     p.requires_grad = False
#             #                 break
#
#         # 加载预训练模型
#         load_path = self.opt['path'].get('pretrain_network_g', None)
#         # print('#########################################################################',load_path)
#         logger = get_root_logger()
#         if load_path is not None:
#             logger.info(f'Loading net_g from {load_path}')
#             self.load_network(self.net_g, load_path, self.opt['path']['strict_load'])
#
#         if self.is_train:
#             self.init_training_settings()
#             # self.use_dis = (self.opt['train']['gan_opt']['loss_weight'] != 0)
#             # self.net_d_best = copy.deepcopy(self.net_d)
#
#         self.net_g_best = copy.deepcopy(self.net_g)
#
#     def init_training_settings(self):
#         logger = get_root_logger()
#         train_opt = self.opt['train']
#         self.net_g.train()
#
#         # define network net_d
#         # self.net_d = build_network(self.opt['network_d'])
#         # self.net_d = self.model_to_device(self.net_d)
#         # load pretrained d models
#         # load_path = self.opt['path'].get('pretrain_network_d', None)
#         # # print(load_path)
#         # if load_path is not None:
#         #     logger.info(f'Loading net_d from {load_path}')
#         #     self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True))
#
#         # self.net_d.train()
#
#         # 定义损失函数
#         if train_opt.get('pixel_opt'):
#             self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
#         else:
#             self.cri_pix = None
#
#         if train_opt.get('fft_opt'):
#             self.cri_fft = build_loss(train_opt['fft_opt']).to(self.device)
#         else:
#             self.cri_fft = None
#
#         # if train_opt.get('perceptual_opt'):
#         #     self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
#         #     self.model_to_device(self.cri_perceptual)
#         # else:
#         #     self.cri_perceptual = None
#
#         # if train_opt.get('gan_opt'):
#         #     self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)
#
#         # self.net_d_iters = train_opt.get('net_d_iters', 1)
#         # self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)
#
#         # set up optimizers and schedulers
#         self.setup_optimizers()
#         self.setup_schedulers()
#
#     def setup_optimizers(self):
#         train_opt = self.opt['train']
#         optim_params = []
#         for k, v in self.net_g.named_parameters():
#             optim_params.append(v)
#             if not v.requires_grad:
#                 logger = get_root_logger()
#                 logger.warning(f'Params {k} will not be optimized.')
#
#         # 定义优化器
#         optim_type = train_opt['optim_g'].pop('type')
#         optim_class = getattr(torch.optim, optim_type)
#         self.optimizer_g = optim_class(optim_params, **train_opt['optim_g'])
#         self.optimizers.append(self.optimizer_g)
#
#         # optimizer d
#         # optim_type = train_opt['optim_d'].pop('type')
#         # optim_class = getattr(torch.optim, optim_type)
#         # self.optimizer_d = optim_class(self.net_d.parameters(), **train_opt['optim_d'])
#         # self.optimizers.append(self.optimizer_d)
#
#     def feed_data(self, data):
#         self.lq = data['lq'].to(self.device)
#         # self.lq_equalize = data['lq_equalize'].to(self.device)
#
#         if 'gt' in data:
#             self.gt = data['gt'].to(self.device)
#
#
#     def print_network(self, model):
#         num_params = 0
#         for p in model.parameters():
#             num_params += p.numel()
#         print(model)
#         print("The number of parameters: {}".format(num_params))
#
#     def optimize_parameters(self, current_iter):
#         train_opt = self.opt['train']
#
#         # for p in self.net_d.parameters():
#         #     p.requires_grad = False
#         self.optimizer_g.zero_grad()
#
#         self.LL, self.output = self.net_g(self.lq)  # LL, x_final
#
#         # if current_iter==0:
#
#         l_g_total = 0
#         loss_dict = OrderedDict()
#
#         dwt = DWT()
#         n, c, h, w = self.gt.shape
#
#         gt_dwt = dwt(self.gt)
#         gt_LL, gt_high0 = gt_dwt[:n, ...], gt_dwt[n:, ...]
#
#
#         l_pix = self.l1(self.output, self.gt) + (1 - self.ssim(self.output, self.gt)) * 0.1
#         l_g_total += l_pix
#         loss_dict['l_pix'] = l_pix
#
#         l_LL = self.l1(self.LL, gt_LL)
#         l_g_total += l_LL
#         loss_dict['l_LL'] = l_LL
#
#         l_cr = self.cr(self.output, self.gt, self.lq) # self.cr(self.output, self.gt, self.x_stage1)
#         l_g_total += 0.01 * l_cr
#         loss_dict['l_cr'] = l_cr
#
#         if train_opt.get('fft_opt', None):
#             l_fft = self.cri_fft(self.output, self.gt)
#             l_g_total += 0.1 * l_fft
#             loss_dict['l_freq'] = l_fft
#
#         l_g_total.mean().backward()
#
#         self.optimizer_g.step()
#
#         self.log_dict = self.reduce_loss_dict(loss_dict)
#
#     def test(self):
#         self.net_g.eval()
#         net_g = self.get_bare_model(self.net_g)
#         min_size = 8000 * 8000  # use smaller min_size with limited GPU memory
#         lq_input = self.lq
#         # restoration = self.net_g(self.lq)
#         _, _, h, w = lq_input.shape
#         if h * w < min_size:
#             # out_img, feature_degradation, self.output = self.net_g(self.lq, feature=feature_degradation)
#             self.output = net_g.test(lq_input)
#         else:
#             self.output = net_g.test_tile(lq_input)
#         self.net_g.train()
#
#     def dist_validation(self, dataloader, current_iter, epoch, tb_logger, save_img, save_as_dir=None):
#         logger = get_root_logger()
#         logger.info('Only support single GPU validation.')
#         self.nondist_validation(dataloader, current_iter, epoch, tb_logger, save_img, save_as_dir)
#
#     def nondist_validation(self, dataloader, current_iter, epoch, tb_logger,
#                            save_img, save_as_dir):
#         # dataset_name = dataloader.dataset.opt['name']
#         dataset_name = 'NTIRE2024'
#         with_metrics = self.opt['val'].get('metrics') is not None
#         if with_metrics:
#             self.metric_results = {
#                 metric: 0
#                 for metric in self.opt['val']['metrics'].keys()
#             }
#
#         pbar = tqdm(total=len(dataloader), unit='image')
#
#         if with_metrics:
#             if not hasattr(self, 'metric_results'):  # only execute in the first run
#                 self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
#             # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
#             self._initialize_best_metric_results(dataset_name)
#
#             # zero self.metric_results
#             self.metric_results = {metric: 0 for metric in self.metric_results}
#             self.key_metric = self.opt['val'].get('key_metric')
#
#         for idx, val_data in enumerate(dataloader):
#             img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
#             self.feed_data(val_data)
#             self.test()
#
#             sr_img = tensor2img(self.output)
#             metric_data = [img2tensor(sr_img).unsqueeze(0) / 255, self.gt]
#
#             # tentative for out of GPU memory
#             del self.lq
#             del self.output
#             torch.cuda.empty_cache()
#
#             if save_img:
#                 if self.opt['is_train']:
#                     save_img_path = osp.join(self.opt['path']['visualization'], 'image_results',
#                                              f'{current_iter}',
#                                              f'{img_name}.png')
#                 else:
#                     if self.opt['val']['suffix']:
#                         save_img_path = osp.join(
#                             self.opt['path']['visualization'], dataset_name,
#                             f'{img_name}_{self.opt["val"]["suffix"]}.png')
#                     else:
#                         save_img_path = osp.join(
#                             self.opt['path']['visualization'], dataset_name,
#                             f'{img_name}_{self.opt["name"]}.png')
#                 if save_as_dir:
#                     save_as_img_path = osp.join(save_as_dir, f'{img_name}.png')
#                     imwrite(sr_img, save_as_img_path)
#                 imwrite(sr_img, save_img_path)
#
#             if with_metrics:
#                 # calculate metrics
#                 for name, opt_ in self.opt['val']['metrics'].items():
#                     tmp_result = self.metric_funcs[name](*metric_data)
#                     self.metric_results[name] += tmp_result.item()
#
#             pbar.update(1)
#             pbar.set_description(f'Test {img_name}')
#
#         pbar.close()
#
#         if with_metrics:
#             # calculate average metric
#             for metric in self.metric_results.keys():
#                 self.metric_results[metric] /= (idx + 1)
#
#             if self.key_metric is not None:
#                 # If the best metric is updated, update and save best model
#                 to_update = self._update_best_metric_result(dataset_name, self.key_metric,
#                                                             self.metric_results[self.key_metric], current_iter)
#
#                 if to_update:
#                     for name, opt_ in self.opt['val']['metrics'].items():
#                         self._update_metric_result(dataset_name, name, self.metric_results[name], current_iter)
#                     self.copy_model(self.net_g, self.net_g_best)
#                     # self.copy_model(self.net_d, self.net_d_best)
#                     self.save_network(self.net_g, 'net_g_best', current_iter, epoch)
#                     # self.save_network(self.net_d, 'net_d_best', current_iter, epoch)
#             else:
#                 # update each metric separately
#                 updated = []
#                 for name, opt_ in self.opt['val']['metrics'].items():
#                     tmp_updated = self._update_best_metric_result(dataset_name, name, self.metric_results[name],
#                                                                   current_iter)
#                     updated.append(tmp_updated)
#                 # save best model if any metric is updated
#                 if sum(updated):
#                     self.copy_model(self.net_g, self.net_g_best)
#                     # self.copy_model(self.net_d, self.net_d_best)
#                     self.save_network(self.net_g, 'net_g_best', '')
#                     # self.save_network(self.net_d, 'net_d_best', '')
#
#             self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
#
#     def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
#         log_str = f'Validation {dataset_name}\n'
#         for metric, value in self.metric_results.items():
#             log_str += f'\t # {metric}: {value:.4f}'
#             if hasattr(self, 'best_metric_results'):
#                 log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
#                             f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
#             log_str += '\n'
#
#         logger = get_root_logger()
#         logger.info(log_str)
# #        if tb_logger:
# #            for metric, value in self.metric_results.items():
# #                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)
#
#     def vis_single_code(self, up_factor=2):
#         net_g = self.get_bare_model(self.net_g)
#         codenum = self.opt['network_g']['codebook_params'][0][1]
#         with torch.no_grad():
#             code_idx = torch.arange(codenum).reshape(codenum, 1, 1, 1)
#             code_idx = code_idx.repeat(1, 1, up_factor, up_factor)
#             output_img = net_g.decode_indices(code_idx)
#             output_img = tvu.make_grid(output_img, nrow=32)
#
#         return output_img.unsqueeze(0)
#
#     def get_current_visuals(self):
#         vis_samples = 16
#         out_dict = OrderedDict()
#         out_dict['lq'] = self.lq.detach().cpu()[:vis_samples]
#         out_dict['result'] = self.output.detach().cpu()[:vis_samples]
#         if hasattr(self, 'gt'):
#             out_dict['gt'] = self.gt.detach().cpu()[:vis_samples]
#         return out_dict
#
#     def save(self, epoch, current_iter):
#         self.save_network(self.net_g, 'net_g', current_iter, epoch)
#         # self.save_network(self.net_d, 'net_d', current_iter, epoch)
#         self.save_training_state(epoch, current_iter)