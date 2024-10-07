import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numbers
from einops import rearrange
from ipdb import set_trace as st
 
##---------- SM Block ---------- 
class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5*x*(1+F.tanh(np.sqrt(2/np.pi)*(x+0.044715*torch.pow(x,3))))


def gelu(x):
    return 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*np.power(x,3))))

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
 
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        'define my gelu'
        self.gelu = GELU()

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        # x = F.gelu(x1) * x2
        x = self.gelu(x1) * x2
        x = self.project_out(x)
        return x

 
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        kv = self.kv_dwconv(self.kv(x))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(y))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

 
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=2, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):

        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, refl, ill_feat):  

         
        ill = ill_feat
        refl = self.norm1(refl)
        ill = self.norm1(ill)
        refl = refl + self.attn(refl, ill)
        refl = refl + self.ffn(self.norm2(refl))
        refl = torch.mul(refl, ill_feat)

        return refl
    
 
##---------- Selective Kernel Feature Fusion (SKFF) ----------
class SKFF(nn.Module):
    def __init__(self, in_channels, height=3,reduction=8,bias=False):
        super(SKFF, self).__init__()
        
        self.height = height
        d = max(int(in_channels/reduction),4)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.LeakyReLU(0.2))

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1,bias=bias))
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats =  inp_feats[0].shape[1]
        
        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])
        
        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        # stx()
        attention_vectors = self.softmax(attention_vectors)
        
        feats_V = torch.sum(inp_feats*attention_vectors, dim=1)
        
        return feats_V     

 
##---------- Context Block ----------    
class ContextBlock(nn.Module):
    
    def __init__(self, n_feat, bias=False):
        super(ContextBlock, self).__init__()
        
        self.head = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=3, bias=bias, padding=1, groups=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_feat, n_feat, kernel_size=3, bias=bias, padding=1, groups=2)
        )

        self.conv_mask = nn.Conv2d(n_feat, 1, kernel_size=1, bias=bias)
        self.softmax = nn.Softmax(dim=2)

        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        )
        
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def modeling(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(3)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def forward(self, x):
        # [N, C, H, W]
        inp = x
        inp = self.head(inp)
        
        # [N, C, 1, 1]
        context = self.modeling(inp)

        # [N, C, 1, 1]
        channel_add_term = self.channel_add_conv(context)
        inp = inp + channel_add_term
        x = x + self.act(inp)

        return x
 
### --------- Residual Context Block (RCB) ----------
class RCBdown(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=8, bias=False, groups=1):
        super(RCBdown, self).__init__()
        
        act = nn.LeakyReLU(0.2)

        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups),
            act, 
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups),
            act
        )

        self.act = act
        
        self.gcnet = ContextBlock(n_feat, bias=bias)

    def forward(self, x):
        inp = x
        x = self.body(x)
        res = self.act(self.gcnet(x))
        res += inp
        return res
    
    
class RCBup(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=8, bias=False, groups=1):
        super(RCBup, self).__init__()
        
        act = nn.LeakyReLU(0.2)
        
        self.body_head = nn.Sequential(
            nn.Conv2d(2*n_feat, n_feat, kernel_size=1, stride=1, bias=bias, groups=groups),
            act
        )

        self.body = nn.Sequential(
            nn.Conv2d(2*n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups),
            act, 
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups),
            act
        )

        self.act = act
        
        self.gcnet = ContextBlock(n_feat, bias=bias)

    def forward(self, x):
        inp = self.body_head(x)
        x = self.body(x)
        res = self.act(self.gcnet(x))
        res += inp
        return res
    
 
##---------- Resizing Modules ----------    
class Down(nn.Module):
    def __init__(self, in_channels, chan_factor, bias=False):
        super(Down, self).__init__()

        self.bot = nn.Sequential(
            nn.AvgPool2d(2, ceil_mode=True, count_include_pad=False),
            nn.Conv2d(in_channels, int(in_channels*chan_factor), 1, stride=1, padding=0, bias=bias)
            )

    def forward(self, x):
        return self.bot(x)

class DownSample(nn.Module):
    def __init__(self, in_channels, scale_factor, chan_factor=2, kernel_size=3):
        super(DownSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(Down(in_channels, chan_factor))
            in_channels = int(in_channels * chan_factor)
        
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, chan_factor, bias=False):
        super(Up, self).__init__()
        # nn.Conv2d(in_channels, int(in_channels//chan_factor), 1, stride=1, padding=0, bias=bias),
        self.bot = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=bias),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias)
            )

    def forward(self, x):
        return self.bot(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, scale_factor, chan_factor=2, kernel_size=3):
        super(UpSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(Up(in_channels, chan_factor))
            in_channels = int(in_channels // chan_factor)
        
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x
    
 
##-------------------- MSCRetinexNet  -----------------------

##-------------------- Decomposition  -----------------------
class Decomposition(nn.Module):
    def __init__(self, bias=False):
        super(Decomposition, self).__init__()

        self.lrelu = nn.LeakyReLU(0.2, inplace=False)
        self.conv_in = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1)
        
        self.conv1 = RCBdown(n_feat=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = RCBdown(n_feat=64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = RCBdown(n_feat=64)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.conv4 = RCBdown(n_feat=64)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        self.conv5 = RCBdown(n_feat=64)
        
        self.upv6 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.conv6 = RCBup(n_feat=64)
        
        self.upv7 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.conv7 = RCBup(n_feat=64)
        
        self.upv8 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.conv8 = RCBup(n_feat=64)
        
        self.upv9 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.conv9 = RCBup(n_feat=64)
        
        self.conv10_1 = nn.Conv2d(64, 4, kernel_size=1, stride=1)
        
    def forward(self, x):
        input_max= torch.max(x, dim=1, keepdim=True)[0]
        x = torch.cat((x, input_max), dim=1)
        
        conv1 = self.lrelu(self.conv_in(x))
        
        conv1 = self.conv1(conv1)
        pool1 = self.pool1(conv1)
        
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        
        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        
        conv5 = self.conv5(pool4)
        
        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.conv6(up6)
        
        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.conv7(up7)
        
        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.conv8(up8)
        
        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.conv9(up9)
        
        conv10 = self.conv10_1(conv9)
        R = torch.sigmoid(conv10[:, 0:3, :, :])
        L = torch.sigmoid(conv10[:, 3:4, :, :])
        # st()
        return L,R
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)


##--------------------  Illumination  Encoder  -----------------------
class Illumination(nn.Module):
    def __init__(self, inp_channels=1, out_channels=1, n_feat=64, scale=1, bias=False):
        super(Illumination, self).__init__()
        
        self.scale = scale
        
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)
        self.conv_in = nn.Conv2d(inp_channels, n_feat, kernel_size=3, stride=1, padding=1)
        
        self.conv1 = RCBdown(n_feat=n_feat)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = RCBdown(n_feat=n_feat)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = RCBdown(n_feat=n_feat)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.conv4 = RCBdown(n_feat=n_feat)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        self.conv5 = RCBdown(n_feat=n_feat)
        
        self.upv6 = nn.ConvTranspose2d(n_feat, n_feat, 2, stride=2)
        self.conv6 = RCBup(n_feat=n_feat)
        
        self.upv7 = nn.ConvTranspose2d(n_feat, n_feat, 2, stride=2)
        self.conv7 = RCBup(n_feat=n_feat)
        
        self.upv8 = nn.ConvTranspose2d(n_feat, n_feat, 2, stride=2)
        self.conv8 = RCBup(n_feat=n_feat)
        
        self.upv9 = nn.ConvTranspose2d(n_feat, n_feat, 2, stride=2)
        self.conv9 = RCBup(n_feat=n_feat)
        
    def forward(self, refl):
    
        
        conv1 = self.lrelu(self.conv_in(refl))
        
        conv1 = self.conv1(conv1)
        pool1 = self.pool1(conv1)
        
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        
        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        
        conv5 = self.conv5(pool4)
        
        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.conv6(up6)
        
        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.conv7(up7)
        
        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.conv8(up8)
        
        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.conv9(up9)
        
        return [conv5, conv6, conv7, conv8, conv9]
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)
                
                
##--------------------  Reflectance  Encoder  -----------------------
class Reflectance(nn.Module):
    def __init__(self, inp_channels=3, out_channels=3, n_feat=64, scale=1, bias=False):
        super(Reflectance, self).__init__()
        
        self.scale = scale
        
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)
        self.conv_in = nn.Conv2d(inp_channels, n_feat, kernel_size=3, stride=1, padding=1)
        
        self.conv1 = RCBdown(n_feat=n_feat)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = RCBdown(n_feat=n_feat)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = RCBdown(n_feat=n_feat)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.conv4 = RCBdown(n_feat=n_feat)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        self.conv5 = RCBdown(n_feat=n_feat)
        self.sm5 = TransformerBlock(dim=n_feat)
        
        self.upv6 = nn.ConvTranspose2d(n_feat, n_feat, 2, stride=2)
        self.conv6 = RCBup(n_feat=n_feat)
        self.sm6 = TransformerBlock(dim=n_feat)
        
        self.upv7 = nn.ConvTranspose2d(n_feat, n_feat, 2, stride=2)
        self.conv7 = RCBup(n_feat=n_feat)
        self.sm7 = TransformerBlock(dim=n_feat)
        
        self.upv8 = nn.ConvTranspose2d(n_feat, n_feat, 2, stride=2)
        self.conv8 = RCBup(n_feat=n_feat)
        self.sm8 = TransformerBlock(dim=n_feat)
        
        self.upv9 = nn.ConvTranspose2d(n_feat, n_feat, 2, stride=2)
        self.conv9 = RCBup(n_feat=n_feat)
        self.sm9 = TransformerBlock(dim=n_feat)
        
    def forward(self, refl, ill_fea):
        
        conv1 = self.lrelu(self.conv_in(refl))
        
        conv1 = self.conv1(conv1)
        pool1 = self.pool1(conv1)
        
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        
        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        
        conv5 = self.conv5(pool4)
        conv5 = self.sm5(conv5, ill_fea[0])
        
        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.conv6(up6)
        conv6 = self.sm6(conv6, ill_fea[1])
        
        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.conv7(up7)
        conv7 = self.sm7(conv7, ill_fea[2])
        
        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.conv8(up8)
        conv8 = self.sm8(conv8, ill_fea[3])
        
        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.conv9(up9)
        conv9 = self.sm9(conv9, ill_fea[4])
                            
        return [conv5, conv6, conv7, conv8, conv9]
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)
                
                
##--------------------  Fusion  Block  -----------------------
class MRFSRB(nn.Module):
    def __init__(self, n_feat, bias, groups, out_channels=3, scale=1):
        super(MRFSRB, self).__init__()
        self.scale = scale
        
        self.dau_top = RCBdown(n_feat, bias=bias, groups=groups)
        self.dau_midtop = RCBdown(n_feat, bias=bias, groups=groups)
        self.dau_midmid = RCBdown(n_feat, bias=bias, groups=groups)
        self.dau_midbot = RCBdown(n_feat, bias=bias, groups=groups)
        self.dau_bot = RCBdown(n_feat, bias=bias, groups=groups)
        
        self.up21_1 = UpSample(n_feat, 2)
        self.up21_2 = UpSample(n_feat, 2)
        self.up32_1 = UpSample(n_feat, 2)
        self.up32_2 = UpSample(n_feat, 2)
        self.up41_1 = UpSample(n_feat, 2)
        self.up41_2 = UpSample(n_feat, 2)
        self.up52_1 = UpSample(n_feat, 2)
        self.up52_2 = UpSample(n_feat, 2)
        
        self.conv_out = nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0, bias=bias)
        
 
        self.skff_top = SKFF(n_feat, 2)
        self.skff_midtop = SKFF(n_feat, 2)
        self.skff_midmid = SKFF(n_feat, 2)
        self.skff_midbot = SKFF(n_feat, 2)
        
        # upsample
        if scale == 2:
            self.conv_up_r1 = nn.Conv2d(n_feat, n_feat, 3, 1, 1)
        if scale == 4:
            self.conv_up_r1 = nn.Conv2d(n_feat, n_feat, 3, 1, 1)
            self.conv_up_r2 = nn.Conv2d(n_feat, n_feat, 3, 1, 1)
        
        self.conv_out_r = nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
    def forward(self, inp):
        x_top = inp[4].clone()
        x_midtop = inp[3].clone()
        x_midmid = inp[2].clone()
        x_midbot = inp[1].clone()
        x_bot = inp[0].clone()
        
        x_top = self.dau_top(x_top)
        x_midtop = self.dau_midtop(x_midtop)
        x_midmid = self.dau_midmid(x_midmid)
        x_midbot = self.dau_midbot(x_midbot)
        x_bot = self.dau_bot(x_bot)

        x_midbot = self.skff_midbot([x_midbot, self.up52_1(x_bot)])
        x_midmid = self.skff_midmid([x_midmid, self.up41_1(x_midbot)])
        x_midtop = self.skff_midtop([x_midtop, self.up32_1(x_midmid)])
        x_top = self.skff_top([x_top, self.up21_1(x_midtop)])
        
        x_top = self.dau_top(x_top)
        x_midtop = self.dau_midtop(x_midtop)
        x_midmid = self.dau_midmid(x_midmid)
        x_midbot = self.dau_midbot(x_midbot)
        x_bot = self.dau_bot(x_bot)
        
        x_midbot = self.skff_midbot([x_midbot, self.up52_2(x_bot)])
        x_midmid = self.skff_midmid([x_midmid, self.up41_2(x_midbot)])
        x_midtop = self.skff_midtop([x_midtop, self.up32_2(x_midmid)])
        x_top = self.skff_top([x_top, self.up21_2(x_midtop)])
        
        out = self.conv_out(x_top)
        out = out + inp[4]
        
        if self.scale == 2:
            out = self.lrelu(self.conv_up_r1(F.interpolate(out, scale_factor=2, mode='nearest')))
        if self.scale == 4:
            out = self.lrelu(self.conv_up_r1(F.interpolate(out, scale_factor=2, mode='nearest')))
            out = self.lrelu(self.conv_up_r2(F.interpolate(out, scale_factor=2, mode='nearest')))
        out = torch.sigmoid(self.conv_out_r(out))

        return out
    
    
##--------------------  MSCRetinexSRNet  -----------------------
class MSCRetinexSRNetv12(nn.Module):
    def __init__(self,
        inp_channels=3,
        out_channels=3,
        n_feat=64,
        scale=1,
        bias=False
    ):
        super(MSCRetinexSRNetv12, self).__init__()
        
        self.n_feat = n_feat
        self.scale = scale

        self.decomposition = Decomposition()  
        self.illumination = Illumination(1, 1, n_feat, scale)        
        self.reflectance = Reflectance(inp_channels, out_channels, n_feat, scale)

        self.mrfsr_ill = MRFSRB(n_feat=n_feat, bias=bias, groups=4, out_channels=1, scale=scale)
        self.mrfsr_refl = MRFSRB(n_feat=n_feat, bias=bias, groups=4, out_channels=out_channels, scale=scale)
        
    def forward(self, inp_img_lllr, inp_img_nlhr=None, trainer=False):
        lllr_ill, lllr_refl = self.decomposition(inp_img_lllr)
        
        if trainer:            
            nlhr_ill, nlhr_refl = self.decomposition(inp_img_nlhr)            
        
        ill_fea = self.illumination(lllr_ill)        
        refl_fea = self.reflectance(lllr_refl, ill_fea)        
        
        nlsr_ill = self.mrfsr_ill(ill_fea)
        nlsr_ill3 = torch.cat((nlsr_ill, nlsr_ill, nlsr_ill), dim=1)
        
        nlsr_refl = self.mrfsr_refl(refl_fea)       

        img_nlsr = torch.mul(nlsr_refl, nlsr_ill3)
        
        if trainer:
            return img_nlsr, lllr_ill, lllr_refl, nlhr_ill, nlhr_refl, nlsr_ill, nlsr_refl
        else:
            return img_nlsr, nlsr_ill, nlsr_refl
        
        
if __name__== '__main__':
    inp = torch.randn((1, 3, 624, 624))
    gt = torch.randn((1, 3, 1248, 1248))
    net = MSCRetinexSRNetv12(scale=2)
    img_nlsr, lllr_ill, lllr_refl, nlhr_ill, nlhr_refl, nlsr_ill, nlsr_refl = net(inp, gt, True)
 
        
 