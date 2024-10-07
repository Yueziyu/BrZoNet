import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
from torchvision import transforms
from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

import os
import random
import numpy as np
import cv2
import torch.nn.functional as F
from functools import partial
from ipdb import set_trace as st


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])  
class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1,1)).item()
    
        r_index = torch.randperm(target.size(0)).to(self.device)
    
        target = lam * target + (1-lam) * target[r_index, :]
        input_ = lam * input_ + (1-lam) * input_[r_index, :]
    
        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments)-1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_
    
class Mixing_Augment3:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_, lh):
        lam = self.dist.rsample((1,1)).item()
    
        r_index = torch.randperm(target.size(0)).to(self.device)
    
        target = lam * target + (1-lam) * target[r_index, :]
        input_ = lam * input_ + (1-lam) * input_[r_index, :]
        lh = lam * lh + (1-lam) * lh[r_index, :]
    
        return target, input_, lh

    def __call__(self, target, input_, lh):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_, lh = self.augments[augment](target, input_, lh)
        else:
            augment = random.randint(0, len(self.augments)-1)
            target, input_, lh = self.augments[augment](target, input_, lh)
        return target, input_, lh


 
class MSCRetinexSRModelv1(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(MSCRetinexSRModelv1, self).__init__(opt)

        # define network

        self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
        if self.mixing_flag:
            mixup_beta       = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
            use_identity     = self.opt['train']['mixing_augs'].get('use_identity', False)
            self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = define_network(self.opt['network_g']).to(
                self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt['path'].get('strict_load_g',
                                                       True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(self.device)
        else:
            raise ValueError('pixel loss are None.')
            
        if train_opt.get('SATV_opt'):
            satv_type = train_opt['SATV_opt'].pop('type')
            cri_SATV_cls = getattr(loss_module, satv_type)
            self.cri_SATV = cri_SATV_cls(**train_opt['SATV_opt']).to(self.device)
            
        if train_opt.get('perceptual_opt'):
            perceptual_type = train_opt['perceptual_opt'].pop('type')
            cri_perceptual_cls = getattr(loss_module, perceptual_type)
            self.cri_perceptual = cri_perceptual_cls(**train_opt['perceptual_opt']).to(self.device)
            
        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
            
        if 'gray' in data:
            self.atten = data['gray'].to(self.device)

        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)
        

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
            
        if 'gray' in data:
            self.atten = data['gray'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        # if self.opt['datasets']['train']['use_grayatten']:
        if self.opt['use_grayatten']:
            preds, lllr_ills, lllr_refls, nlhr_ills, nlhr_refls, nlsr_ills, nlsr_refls = self.net_g(self.lq, self.atten, self.gt, True)
        else:
            preds, lllr_ills, lllr_refls, nlhr_ills, nlhr_refls, nlsr_ills, nlsr_refls = self.net_g(self.lq, self.gt, True)
        # st()
        if not isinstance(preds, list):
            preds = [preds]
        if not isinstance(lllr_ills, list):
            lllr_ills = [lllr_ills]
        if not isinstance(lllr_refls, list):
            lllr_refls = [lllr_refls]
        if not isinstance(nlhr_ills, list):
            nlhr_ills = [nlhr_ills]
        if not isinstance(nlhr_refls, list):
            nlhr_refls = [nlhr_refls]
        if not isinstance(nlsr_ills, list):
            nlsr_ills = [nlsr_ills]
        if not isinstance(nlsr_refls, list):
            nlsr_refls = [nlsr_refls]

        self.output = preds[-1]
        # st()

        loss_dict = OrderedDict()
        # pixel loss
        l_pix_lllr = 0.
        l_pix_nlhr = 0.
        l_pix_nlsr = 0.
        l_pix_sr_ill = 0.
        l_pix_sr_refl = 0.
        l_SATV_lllr = 0.
        l_SATV_nlhr = 0.
        l_SATV_nlsr = 0.
        l_g_percep_nlsr = 0.
        l_g_style_nlsr = 0.
        
        for pred in preds:
            l_pix_nlsr += self.cri_pix(pred, self.gt)
            l_g_percep, l_g_style = self.cri_perceptual(pred, self.gt)
            if l_g_percep is not None:
                l_g_percep_nlsr += l_g_percep
            if l_g_style is not None:
                l_g_style_nlsr += l_g_style
            
        for lllr_ill, lllr_refl in zip(lllr_ills, lllr_refls):
            l_pix_lllr += self.cri_pix(lllr_refl*torch.cat((lllr_ill, lllr_ill, lllr_ill), dim=1), self.lq)
            l_SATV_lllr += self.cri_SATV(lllr_ill, lllr_refl)
            
        for nlhr_ill, nlhr_refl in zip(nlhr_ills, nlhr_refls):
            l_pix_nlhr += self.cri_pix(nlhr_refl*torch.cat((nlhr_ill, nlhr_ill, nlhr_ill), dim=1), self.gt)
        
        for nlsr_ill, nlhr_ill in zip(nlsr_ills, nlhr_ills):
            l_pix_sr_ill += self.cri_pix(nlsr_ill, nlhr_ill)
            
        for nlsr_refl, nlhr_refl in zip(nlsr_refls, nlhr_refls):
            l_pix_sr_refl += self.cri_pix(nlsr_refl, nlhr_refl)
            
        for nlsr_ill, nlsr_refl in zip(nlsr_ills, nlsr_refls):
            l_SATV_nlsr += self.cri_SATV(nlsr_ill, nlsr_refl)
        
        for nlhr_ill, nlhr_refl in zip(nlhr_ills, nlhr_refls):
            l_SATV_nlhr += self.cri_SATV(nlhr_ill, nlhr_refl)

        l_total = l_pix_nlsr + l_pix_lllr + 0.5*l_pix_sr_refl + 0.5*l_pix_sr_ill + 0.8*l_SATV_lllr + 0.8*l_SATV_nlsr \
                + l_pix_nlhr + 0.8*l_SATV_nlhr + l_g_percep_nlsr

        loss_dict['l_pix_nlsr'] = l_pix_nlsr
        loss_dict['l_pix_lllr'] = l_pix_lllr
        loss_dict['l_pix_nlhr'] = l_pix_nlhr
        loss_dict['l_pix_sr_ill'] = l_pix_sr_ill
        loss_dict['l_pix_sr_refl'] = l_pix_sr_refl
        loss_dict['l_SATV_lllr'] = l_SATV_lllr
        loss_dict['l_SATV_nlhr'] = l_SATV_nlhr
        loss_dict['l_SATV_nlsr'] = l_SATV_nlsr
        loss_dict['l_g_percep_nlsr'] = l_g_percep_nlsr
        loss_dict['l_total'] = l_total

        # l_pix.backward()
        l_total.backward()
        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def pad_test(self, window_size):        
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        # if self.opt['datasets']['train']['use_grayatten']:
        if self.opt['use_grayatten']:
            gray_atten = F.pad(self.atten, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        else:
            gray_atten = None
        self.nonpad_test(img, gray_atten)
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img=None, gray_atten=None):
        if img is None:
            img = self.lq    
        # if gray_atten is None and self.opt['datasets']['train']['use_grayatten']:
        if gray_atten is None and self.opt['use_grayatten']:
            gray_atten = self.atten  
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                # if self.opt['datasets']['train']['use_grayatten']:
                if self.opt['use_grayatten']:
                    pred, ill, refl = self.net_g_ema(img, gray_atten, trainer=False)
                else:
                    pred, ill, refl = self.net_g_ema(img, trainer=False)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
        else:
            self.net_g.eval()
            with torch.no_grad():
                # if self.opt['datasets']['train']['use_grayatten']:
                if self.opt['use_grayatten']:
                    pred, ill, refl = self.net_g(img, gray_atten, trainer=False)
                else:
                    pred, ill, refl = self.net_g(img, trainer=False)
            if isinstance(pred, list):
                pred = pred[-1]
            if isinstance(ill, list):
                ill = ill[-1]
            if isinstance(refl, list):
                refl = refl[-1]
            self.output = pred
            self.ill = ill
            self.refl = refl
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        # pbar = tqdm(total=len(dataloader), unit='image')

        window_size = self.opt['val'].get('window_size', 0)

        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test

        cnt = 0

        for idx, val_data in enumerate(tqdm(dataloader)):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data)
            test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            ill_img = tensor2img([visuals['ill']], rgb2bgr=rgb2bgr)
            refl_img = tensor2img([visuals['refl']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            # if self.opt['datasets']['train']['use_grayatten']:
            if self.opt['use_grayatten']:
                del self.atten
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                
                if self.opt['is_train']:
                    
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.png')
                    
                    save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}_gt.png')
                else:
                    
                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}.png')
                    save_ill_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}_ill.png')
                    save_refl_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}_refl.png')
 
                imwrite(sr_img, save_img_path)
 

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)
        return current_metric


    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        out_dict['ill'] = self.ill.detach().cpu()
        out_dict['refl'] = self.refl.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema],
                              'net_g',
                              current_iter,
                              param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)


