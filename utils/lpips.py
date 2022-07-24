############################################################
# The contents below have been combined using files in the #
# following repository:                                    #
# https://github.com/richzhang/PerceptualSimilarity        #
############################################################

from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.nn.functional import normalize
import torch.nn.init as init
from torch.autograd import Variable
import jax
import jax.numpy as jnp
from . import pretrained_networks as pn

from skimage.metrics import structural_similarity as ssim
from skimage import color

import warnings


def normalize_tensor(input_feature, eps=1e-10):
    """ A function to normalize the tensor. """

    norm_factor = torch.sqrt(torch.sum(input_feature ** 2, dim=0, keepdim=True))
    return (input_feature/(norm_factor + eps))


def calculate_l2(p0, p1, range=255.0):
    """ A function to calculate l2 loss. """

    return 0.5 * jnp.mean((p0 / range - p1 / range) ** 2)
 

def calculate_psnr(p0, p1, peak_val=255.0):
    """ A function to calculate Peak signal-to-noise ratio. """

    return 10 * jnp.log10(peak_val ** 2 / jnp.mean((1. * p0 - 1. * p1) ** 2))
  

def calculate_dssim(p0, p1, range=255.0):
    """ A function to calculate structural similarity index. """

    return (1 - ssim(p0, p1, data_range=range, multichannel=True)) / 2.0


def rgb2lab(input_img, mean_cent=False):
    """ A function to convert RGB color space to LAB color space. """

    img_lab = color.rgb2lab(input_img)
    if mean_cent:
      img_lab[:, :, 0] = img_lab[:, :, 0] - 50

    return img_lab


def tensor2numpy(tensor_obj):
	  """ A function to convert a tensor object to numpy array. """

	  return tensor_obj[0].cpu.float().numpy().transpose((1, 2, 0))


def numpy2tensor(numpy_obj):
	  """ A function to convert numpy array to a tensor. """

	  return torch.Tensor(numpy_obj[:,:,:, np.newaxis].transpose((3, 2, 0, 1)))


def tensor2tensorlab(image_tensor, to_norm=True, mc_only=False):
    """ A function to convert tensor to LAB. """

    img = tensor2im(image_tensor)
    img_lab = color.rgb2lab(img)
    if mc_only:
        img_lab[:, :, 0] = img_lab[:, :, 0] - 50
    if to_norm and not mc_only:
        img_lab[:, :, 0] = img_lab[:, :, 0] - 50
        img_lab = img_lab / 100.

    return numpy2tensor(img_lab)


def tensorlab2tensor(lab_tensor, return_inbnd=False):
    """ A function to convert tensor LAB to tensor. """
  
    warnings.filterwarnings("ignore")

    lab = tensor2numpy(lab_tensor) * 100.
    lab[:, :, 0] = lab[:, :, 0] + 50

    rgb_back = 255. * jnp.clip(color.lab2rgb(lab.astype('float')), 0, 1)
    if return_inbnd:
        # convert back to lab, see if we match
        lab_back = color.rgb2lab(rgb_back.astype('uint8'))
        mask = 1. * jnp.isclose(lab_back, lab, atol=2.)
        mask = numpy2tensor(jnp.dot(mask, axis=2)[:, :, jnp.newaxis])
        return im2tensor(rgb_back), mask
    else:
        return im2tensor(rgb_back)


def tensor2im(image_tensor, imtype=jnp.uint8, cent=1., factor=255. / 2.):
    """ A function to convert tensor to image. """

    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (jnp.transpose(image_numpy, (1, 2, 0)) + cent) * factor
    return image_numpy.astype(imtype)


def im2tensor(image, imtype=jnp.uint8, cent=1., factor=255. / 2.):
    """ A function to convert image tensor to tensor. """

    return torch.tensor((image / factor - cent)
                        [:, :, :, jnp.newaxis].transpose((3, 2, 0, 1)))


def tensor2vec(vector_tensor):
    """ A function to convert tensor to vector. """
    return vector_tensor.data.cpu().numpy()[:, :, 0, 0]


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """

    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in jnp.arange(0., 1.1, 0.1):
            if jnp.sum(rec >= t) == 0:
                p = 0
            else:
                p = jnp.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = jnp.concatenate(([0.], rec, [1.]))
        mpre = jnp.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = jnp.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = jnp.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = jnp.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def spatial_average(in_tens, keepdim=True):
    """ A function to calculate spatial average. """

    return in_tens.mean([2, 3], keepdim=keepdim)


def upsample(in_tens, out_HW=(64, 64)):
    """ A function to upsample the image tensor. """
    in_H, in_W = in_tens.shape[2], in_tens.shape[3]
    return nn.Upsample(size=out_HW, mode='bilinear', align_corners=False)(in_tens)


class LPIPS(nn.Module):
    def __init__(self, pretrained=True, net='alex', version='0.1', lpips=True, spatial=False,
                 pnet_rand=False, pnet_tune=False, use_dropout=True, model_path=None, eval_mode=True, verbose=True):
        """ Initializes a perceptual loss torch.nn.Module
        Parameters (default listed first)
        ---------------------------------
        lpips : bool
            [True] use linear layers on top of base/trunk network
            [False] means no linear layers; each layer is averaged together
        pretrained : bool
            This flag controls the linear layers, which are only in effect when lpips=True above
            [True] means linear layers are calibrated with human perceptual judgments
            [False] means linear layers are randomly initialized
        pnet_rand : bool
            [False] means trunk loaded with ImageNet classification weights
            [True] means randomly initialized trunk
        net : str
            ['alex','vgg','squeeze'] are the base/trunk networks available
        version : str
            ['v0.1'] is the default and latest
            ['v0.0'] contained a normalization bug; corresponds to old arxiv v1 (https://arxiv.org/abs/1801.03924v1)
        model_path : 'str'
            [None] is default and loads the pretrained weights from paper https://arxiv.org/abs/1801.03924v1
        The following parameters should only be changed if training the network
        eval_mode : bool
            [True] is for test mode (default)
            [False] is for training mode
        pnet_tune
            [False] keep base/trunk frozen
            [True] tune the base/trunk network
        use_dropout : bool
            [True] to use dropout when training linear layers
            [False] for no dropout when training linear layers
        """

        super(LPIPS, self).__init__()
        if verbose:
            print('Setting up [%s] perceptual loss: trunk [%s], v[%s], spatial [%s]' %
                  ('LPIPS' if lpips else 'baseline', net, version, 'on' if spatial else 'off'))

        self.pnet_type = net
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial
        self.lpips = lpips  # false means baseline of just averaging all layers
        self.version = version
        self.scaling_layer = ScalingLayer()

        if self.pnet_type in ['vgg', 'vgg16']:
            net_type = pn.vgg16
            self.chns = [64, 128, 256, 512, 512]
        elif self.pnet_type == 'alex':
            net_type = pn.alexnet
            self.chns = [64, 192, 384, 256, 256]
        elif self.pnet_type == 'squeeze':
            net_type = pn.squeezenet
            self.chns = [64, 128, 256, 384, 384, 512, 512]
        self.L = len(self.chns)

        self.net = net_type(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune)

        if lpips:
            self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
            self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
            self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
            self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
            self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
            self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
            if self.pnet_type == 'squeeze':  # 7 layers for squeezenet
                self.lin5 = NetLinLayer(self.chns[5], use_dropout=use_dropout)
                self.lin6 = NetLinLayer(self.chns[6], use_dropout=use_dropout)
                self.lins += [self.lin5, self.lin6]
            self.lins = nn.ModuleList(self.lins)

            if pretrained:
                if model_path is None:
                    import inspect
                    import os
                    model_path = os.path.abspath(
                        os.path.join(inspect.getfile(self.__init__), '..', 'weights/v%s/%s.pth' % (version, net)))

                if verbose:
                    print('Loading model from: %s' % model_path)
                self.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)

        if eval_mode:
            self.eval()

    def forward(self, in0, in1, retPerLayer=False, normalize=False):
        if normalize:  # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            in0 = 2 * in0 - 1
            in1 = 2 * in1 - 1

        # v0.0 - original release had a bug, where input was not scaled
        in0_input, in1_input = (self.scaling_layer(in0), self.scaling_layer(in1)) if self.version == '0.1' else (
        in0, in1)
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.L):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        if self.lpips:
            if self.spatial:
                res = [upsample(self.lins[kk](diffs[kk]), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [spatial_average(self.lins[kk](diffs[kk]), keepdim=True) for kk in range(self.L)]
        else:
            if self.spatial:
                res = [upsample(diffs[kk].sum(dim=1, keepdim=True), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [spatial_average(diffs[kk].sum(dim=1, keepdim=True), keepdim=True) for kk in range(self.L)]

        val = res[0]
        for l in range(1, self.L):
            val += res[l]

        if retPerLayer:
            return val, res
        else:
            return val


class ScalingLayer(nn.Module):
    """ A layer which performs scaling operations. """
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout(), ] if use_dropout else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Dist2LogitLayer(nn.Module):
    """ This takes 2 distances, puts through fc layers, spits out value between [0,1] (if use_sigmoid is True) """

    def __init__(self, chn_mid=32, use_sigmoid=True):
        super(Dist2LogitLayer, self).__init__()

        layers = [nn.Conv2d(5, chn_mid, 1, stride=1, padding=0, bias=True), ]
        layers += [nn.LeakyReLU(0.2, True), ]
        layers += [nn.Conv2d(chn_mid, chn_mid, 1, stride=1, padding=0, bias=True), ]
        layers += [nn.LeakyReLU(0.2, True), ]
        layers += [nn.Conv2d(chn_mid, 1, 1, stride=1, padding=0, bias=True), ]
        if use_sigmoid:
            layers += [nn.Sigmoid(), ]
        self.model = nn.Sequential(*layers)

    def forward(self, d0, d1, eps=0.1):
        return self.model.forward(torch.cat((d0, d1, d0 - d1, d0 / (d1 + eps), d1 / (d0 + eps)), dim=1))


class BCERankingLoss(nn.Module):
    def __init__(self, chn_mid=32):
        super(BCERankingLoss, self).__init__()
        self.net = Dist2LogitLayer(chn_mid=chn_mid)
        # self.parameters = list(self.net.parameters())
        self.loss = torch.nn.BCELoss()

    def forward(self, d0, d1, judge):
        per = (judge + 1.) / 2.
        self.logit = self.net.forward(d0, d1)
        return self.loss(self.logit, per)


# L2, DSSIM metrics
class FakeNet(nn.Module):
    def __init__(self, use_gpu=True, colorspace='Lab'):
        super(FakeNet, self).__init__()
        self.use_gpu = use_gpu
        self.colorspace = colorspace


class L2(FakeNet):
    def forward(self, in0, in1, retPerLayer=None):
        assert (in0.size()[0] == 1)  # currently only supports batchSize 1

        if self.colorspace == 'RGB':
            (N, C, X, Y) = in0.size()
            value = torch.mean(torch.mean(torch.mean((in0 - in1) ** 2, dim=1).view(N, 1, X, Y), dim=2).view(N, 1, 1, Y),
                               dim=3).view(N)
            return value
        elif self.colorspace == 'Lab':
            value = calculate_l2(tensor2numpy(tensor2tensorlab(in0.data, to_norm=False)),
                             tensor2numpy(tensor2tensorlab(in1.data, to_norm=False)), range=100.).astype(
                'float')
            ret_var = Variable(torch.Tensor((value,)))
            if self.use_gpu:
                ret_var = ret_var.cuda()
            return ret_var


class DSSIM(FakeNet):

    def forward(self, in0, in1, retPerLayer=None):
        assert (in0.size()[0] == 1)  # currently only supports batchSize 1

        if self.colorspace == 'RGB':
            value = calculate_dssim(1. * tensor2im(in0.data), 1. * tensor2im(in1.data), range=255.).astype(
                'float')
        elif self.colorspace == 'Lab':
            value = calculate_dssim(tensor2numpy(tensor2tensorlab(in0.data, to_norm=False)),
                                tensor2numpy(tensor2tensorlab(in1.data, to_norm=False)), range=100.).astype(
                'float')
        ret_var = Variable(torch.Tensor((value,)))
        if self.use_gpu:
            ret_var = ret_var.cuda()
        return ret_var


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Network', net)
    print('Total number of parameters: %d' % num_params)
