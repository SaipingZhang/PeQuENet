import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
import functools


class QPNET(nn.Module):
    def __init__(self, in_nc=128, nf=64, out_nc=1, base_ks=3):
        """
        Args:
            in_nc: num of input channels.
            nf: num of channels (filters) of each conv layer.
            nb: num of conv layers.
            out_nc: num of output channel. 1 for Y.
            base_ks: kernel size 3x3
        """
        super(QPNET, self).__init__()
        self.one_hot = F.one_hot
        self.fc = nn.Sequential(
            nn.Linear(4, 64),
            nn.Softplus()
        )

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.in_conv = nn.Conv2d(in_nc, nf, base_ks, padding=1)

        self.hid_conv1 = nn.Conv2d(nf, nf, base_ks, padding=1)
        self.hid_conv2 = nn.Conv2d(nf, nf, base_ks, padding=1)

        self.out_conv = nn.Conv2d(nf, out_nc, base_ks, padding=1)

    def forward(self, inputs, qp):
        b, c = qp.size()
        x = F.one_hot(qp, 4)
        x = x.squeeze(1)
        x = x.to(torch.float32)
        x = self.fc(x)
        x = x.view(b, 64, 1, 1)

        out = self.relu(self.in_conv(inputs) * x)
        out1 = self.relu(self.hid_conv1(out) * x)
        out2 = self.relu(self.hid_conv2(out1) * x)
        out = self.out_conv(out2)
        out = self.tanh(out)
        return out


class AttentionBlock(nn.Module):
    def __init__(self, dim, out=False):
        super(AttentionBlock, self).__init__()
        self.embed_dim = dim
        self.proj_tgt = nn.Conv2d(dim, dim // 2, kernel_size=1, padding=0)
        self.out = out
        if self.out:
            self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
        self.proj_ref = nn.Conv2d(dim, dim // 2, kernel_size=1, padding=0)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, tgt, ref, need_weights=False):
        bs, c, h, w = tgt.shape
        att_tgt = self.activation(self.proj_tgt(tgt)).flatten(2).transpose(1, 2)
        att_ref = self.activation(self.proj_ref(ref)).flatten(2)

        ref = ref.flatten(2).transpose(1, 2)
        pre_att = torch.bmm(att_tgt, att_ref)
        att = F.softmax(pre_att, dim=-1)
        fused = torch.bmm(att, ref)
        fused = fused.transpose(1, 2).contiguous().view(bs, c, h, w)
        if self.out:
            att_out = self.proj_out(tgt)
            fused = torch.mul(fused, att_out)
        if need_weights:
            return fused, att.view(bs, h, w, h, w)
        else:
            return fused, None


class DecoderBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = nn.Conv2d(dim_in + dim_out, dim_out, kernel_size=3, padding=1)
        self.activation = nn.ReLU(inplace=True)

    def _upsample(self, x1, x2):
        x1 = self.upsample(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return x

    def forward(self, src, fused1, skip):
        if fused1 is not None:
            src = src + fused1
        x = self.conv1(src)
        x = self.activation(x)
        x = self._upsample(x, skip)
        x = self.conv2(x)
        x = self.activation(x)
        return x


class FeaturesExtractor(nn.Module):
    def __init__(self):
        super(FeaturesExtractor, self).__init__()
        features = vgg19(pretrained=True, progress=False).features
        self.to_relu_1_1 = nn.Sequential()
        self.to_relu_2_1 = nn.Sequential()
        self.to_relu_3_1 = nn.Sequential()
        self.to_relu_4_1 = nn.Sequential()
        self.to_relu_5_1 = nn.Sequential()
        self.to_relu_5_4 = nn.Sequential()

        for x in range(2):
            self.to_relu_1_1.add_module(str(x), features[x])
        for x in range(2, 7):
            self.to_relu_2_1.add_module(str(x), features[x])
        for x in range(7, 12):
            self.to_relu_3_1.add_module(str(x), features[x])
        for x in range(12, 21):
            self.to_relu_4_1.add_module(str(x), features[x])
        for x in range(21, 30):
            self.to_relu_5_1.add_module(str(x), features[x])
        for x in range(30, 36):
            self.to_relu_5_4.add_module(str(x), features[x])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_1(x)
        h_relu_1_1 = h
        h = self.to_relu_2_1(h)
        h_relu_2_1 = h
        h = self.to_relu_3_1(h)
        h_relu_3_1 = h
        h = self.to_relu_4_1(h)
        h_relu_4_1 = h
        h = self.to_relu_5_1(h)
        h_relu_5_1 = h
        h = self.to_relu_5_4(h)
        h_relu_5_4 = h
        out = (h_relu_1_1, h_relu_2_1, h_relu_3_1, h_relu_4_1, h_relu_5_1, h_relu_5_4)
        return out


class PeQuENet(nn.Module):
    def __init__(self, output_nc=1, out=False):
        super(PeQuENet, self).__init__()
        self.features = FeaturesExtractor()
        self.features.eval()

        self.att3_minus = AttentionBlock(256, out)
        self.att4_minus = AttentionBlock(512, out)
        self.att5_minus = AttentionBlock(512, out)

        self.att3_plus = AttentionBlock(256, out)
        self.att4_plus = AttentionBlock(512, out)
        self.att5_plus = AttentionBlock(512, out)

        self.dec4_minus = DecoderBlock(512, 512)
        self.dec3_minus = DecoderBlock(512, 256)
        self.dec2_minus = DecoderBlock(256, 128)
        self.dec1_minus = DecoderBlock(128, 64)

        self.dec4_plus = DecoderBlock(512, 512)
        self.dec3_plus = DecoderBlock(512, 256)
        self.dec2_plus = DecoderBlock(256, 128)
        self.dec1_plus = DecoderBlock(128, 64)

        self.QPNet = QPNET(128,64,1,3)


    def forward(self, radius, inputs, qp):
        #inputs: [BS T H W]
        # features are acquired by a pre-trained VGG19 net
        # h_relu_1_1, h_relu_2_1, h_relu_3_1, h_relu_4_1, h_relu_5_1, h_relu_5_4
        tgt = inputs[:, radius, :, :]
        tgt = torch.unsqueeze(tgt, 1)
        tgt = torch.cat((tgt,tgt,tgt),1)
        ref_minus = inputs[:, radius - 1, :, :]
        ref_minus = torch.unsqueeze(ref_minus, 1)
        ref_minus = torch.cat((ref_minus, ref_minus, ref_minus), 1)
        ref_plus = inputs[:, radius + 1, :, :]
        ref_plus = torch.unsqueeze(ref_plus, 1)
        ref_plus = torch.cat((ref_plus, ref_plus, ref_plus), 1)

        tgt1, tgt2, tgt3, tgt4, tgt5, tgt6 = self.features(tgt)
        _, ref2_minus, ref3_minus, ref4_minus, ref5_minus, _ = self.features(ref_minus)
        _, ref2_plus, ref3_plus, ref4_plus, ref5_plus, _ = self.features(ref_plus)

        fused5_minus = self.att5_minus(tgt5, ref5_minus, need_weights=True)[0]
        fused4_minus = self.att4_minus(tgt4, ref4_minus, need_weights=True)[0]
        fused3_minus = self.att3_minus(tgt3, ref3_minus, need_weights=True)[0]

        fused5_plus = self.att5_plus(tgt5, ref5_plus, need_weights=True)[0]
        fused4_plus = self.att4_plus(tgt4, ref4_plus, need_weights=True)[0]
        fused3_plus = self.att3_plus(tgt3, ref3_plus, need_weights=True)[0]

        dec4_minus = self.dec4_minus(tgt6, fused5_minus, tgt4)
        dec3_minus = self.dec3_minus(dec4_minus, fused4_minus, tgt3)
        dec2_minus = self.dec2_minus(dec3_minus, fused3_minus, tgt2)
        dec1_minus = self.dec1_minus(dec2_minus, None, tgt1)

        dec4_plus = self.dec4_plus(tgt6, fused5_plus, tgt4)
        dec3_plus = self.dec3_plus(dec4_plus, fused4_plus, tgt3)
        dec2_plus = self.dec2_plus(dec3_plus, fused3_plus, tgt2)
        dec1_plus = self.dec1_plus(dec2_plus, None, tgt1)

        fused_feature = torch.cat((dec1_minus,dec1_plus),1)

        pred = self.QPNet(fused_feature,qp)

        return pred
