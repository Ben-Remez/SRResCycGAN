import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

vgg_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
vgg_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class SRResCycGANLosses:
    def __init__(self, vgg):
        self.vgg = vgg
        self.l1_loss = nn.L1Loss()
        self.tv_loss = TVLoss()

    def perceptual_loss(self, sr, hr):
        sr_vgg = self.vgg((sr - vgg_mean.to(sr.device)) / vgg_std.to(sr.device))
        hr_vgg = self.vgg((hr - vgg_mean.to(hr.device)) / vgg_std.to(hr.device))
        return self.l1_loss(sr_vgg, hr_vgg)

    def gan_loss(self, sr_pred, hr_pred):
        return F.binary_cross_entropy_with_logits(sr_pred, torch.ones_like(sr_pred)) + \
               F.binary_cross_entropy_with_logits(hr_pred, torch.zeros_like(hr_pred))

    def total_variation_loss(self, sr):
        return self.tv_loss(sr)

    def content_loss(self, sr, hr):
        return self.l1_loss(sr, hr)

    def cyclic_loss(self, lr_recon, lr):
        return self.l1_loss(lr_recon, lr)

    def total_loss(self, sr, hr, lr_recon, lr, sr_pred, hr_pred):
        # Resize tensors to ensure shape compatibility
        hr_resized = F.interpolate(hr, size=sr.size()[2:], mode='bicubic', align_corners=False)
        lr_resized = F.interpolate(lr, size=lr_recon.size()[2:], mode='bicubic', align_corners=False)

        l_per = self.perceptual_loss(sr, hr_resized)
        l_gan = self.gan_loss(sr_pred, hr_pred)
        l_tv = self.total_variation_loss(sr)
        l_content = self.content_loss(sr, hr_resized)
        l_cyclic = self.cyclic_loss(lr_recon, lr_resized)

        return l_per + l_gan + l_tv + 10 * l_content + 10 * l_cyclic


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


# VGG19 model for perceptual loss
class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        vgg = models.vgg19(pretrained=True)
        self.features = nn.Sequential(*list(vgg.features.children())[:36]).eval()  # Use first 36 layers

        # Freeze parameters
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.features(x)