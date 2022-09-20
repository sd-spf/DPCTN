import torchvision.models.densenet
import torch
import torch.nn as nn
import torch.nn.functional as F

from .CCT import CCT
from .ps.ps_vit import ps_vit_b_18
from .efficientnet_pytorch.model import EfficientNet

def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class dependencymap(nn.Sequential):
    def __init__(self, emb_size: int = 576, n_regions: int = 256, patch_size: int = 1, img_size: int = 48,
                 output_ch: int = 32, cuda=True):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.emb_size = emb_size
        self.output_ch = output_ch
        self.cuda = cuda
        self.outconv = nn.Sequential(
            nn.Conv2d(emb_size, output_ch, kernel_size=1, padding=0),
            nn.BatchNorm2d(output_ch),
            nn.Sigmoid()
        )
        self.out2 = nn.Sigmoid()

    def forward(self, x):

        coeff = torch.zeros((x.size()[0], self.emb_size, self.img_size, self.img_size))

        if self.cuda:
            coeff = coeff.cuda()
        for i in range(0, self.img_size // self.patch_size):
            for j in range(0, self.img_size // self.patch_size):
                value = x[:, (i * self.patch_size) + j]
                value = value.view(value.size()[0], value.size()[1], 1, 1)
                coeff[:, :, self.patch_size * i:self.patch_size * (i + 1),
                self.patch_size * j:self.patch_size * (j + 1)] = value.repeat(1, 1, self.patch_size, self.patch_size)

        global_contexual = self.outconv(coeff)

        return global_contexual



class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - x
        h = self.relu(self.conv2(h))
        return h


class CGRModule(nn.Module):
    def __init__(self, num_in, plane_mid, mids, abn=nn.BatchNorm2d, normalize=False):
        super(CGRModule, self).__init__()

        self.normalize = normalize
        self.num_s = int(plane_mid)
        self.num_n = (mids) * (mids)
        self.priors = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))

        self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_proj = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1, bias=False)

        self.blocker = abn(num_in)

    def forward(self, x, edge):
        edge = F.upsample(edge, (x.size()[-2], x.size()[-1]))

        n, c, h, w = x.size()
        edge = torch.nn.functional.softmax(edge, dim=1)

        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)
        x_proj = self.conv_proj(x)
        x_mask = x_proj * edge
        x_anchor = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)
        x_proj_reshaped = torch.matmul(x_anchor.permute(0, 2, 1), x_proj.reshape(n, self.num_s, -1))
        x_proj_reshaped = torch.nn.functional.softmax(x_proj_reshaped, dim=1)
        x_rproj_reshaped = x_proj_reshaped
        # Project and graph reason
        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))
        x_n_rel = self.gcn(x_n_state)
        # Reproject
        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)  # x_n_rel
        x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])
        out = x + self.blocker(self.conv_extend(x_state))
        return out



class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_relu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_relu(output)

        return output


class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output

class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim/2),1,1,padding=0,bn_acti=True)
        self.bn2 = nn.BatchNorm2d(int(out_dim/2))
        self.conv2 = Conv(int(out_dim/2), int(out_dim/2), 3,1,padding=1, bn_acti=True)
        self.bn3 = nn.BatchNorm2d(int(out_dim/2))
        self.conv3 = Conv(int(out_dim/2), out_dim, 1,1,padding=0, bn_acti=True)
        self.skip_layer = Conv(inp_dim, out_dim,1,1,padding=0, bn_acti=True)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True
        
    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out 

class HA(nn.Module):
   
    def __init__(self, channel):
        super(HA, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = Conv(32, 32, 3,1, padding=1)
        self.conv_upsample2 = Conv(32, 32, 3,1, padding=1)
        self.conv_upsample3 = Conv(32, 32, 3,1, padding=1)
        self.conv_upsample4 = Conv(32, 32, 3,1, padding=1)
        self.conv_upsample5 = Conv(2*32, 2*32, 3,1, padding=1)

        self.conv_concat2 = Conv(2*32, 2*32, 3,1, padding=1)
        self.conv_concat3 = Conv(3*32, 3*32, 3,1, padding=1)
        self.conv4 = Conv(3*32, 3*32, 3,1, padding=1)
        self.conv5 = nn.Conv2d(3*32 , 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


class LA(nn.Module):

    def __init__(self, abn=nn.BatchNorm2d, in_fea=[64, 256, 512], mid_fea=64, out_fea=1):
        super(LA, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_fea[0], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            abn(mid_fea)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_fea[1], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            abn(mid_fea)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_fea[2], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            abn(mid_fea)
        )
        self.conv4 = nn.Conv2d(mid_fea, out_fea, kernel_size=3, padding=1, dilation=1, bias=True)
        self.conv5 = nn.Conv2d(out_fea * 3, out_fea, kernel_size=1, stride=2, padding=0, dilation=1, bias=True)

    def forward(self, x1, x2, x3):
        _, _, h, w = x1.size()

        edge1_fea = self.conv1(x1)
        edge1 = self.conv4(edge1_fea)
        edge2_fea = self.conv2(x2)
        edge2 = self.conv4(edge2_fea)
        edge3_fea = self.conv3(x3)
        edge3 = self.conv4(edge3_fea)

        edge2 = F.interpolate(edge2, size=(h, w), mode='bilinear', align_corners=True)
        edge3 = F.interpolate(edge3, size=(h, w), mode='bilinear', align_corners=True)

        edge = torch.cat([edge1, edge2, edge3], dim=1)
        edge = self.conv5(edge)

        return edge

class TSA(nn.Module):
    def __init__(self, inplanes, planes,size, kernel_size=1, stride=1):
        super(TSA, self).__init__()
        self.a1 = size
        self.inter_a1 = size//2
        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size-1)//2

        self.conv_q_right = nn.Conv2d(self.a1, self.inter_a1, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_v_right = nn.Conv2d(self.a1, self.inter_a1, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_up = nn.Conv2d(self.inter_planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax_right = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.conv_q_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)   #g
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_v_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)   #theta
        self.softmax_left = nn.Softmax(dim=2)

        self.reset_parameters()

    def reset_parameters(self):
        kaiming_init(self.conv_q_right, mode='fan_in')
        kaiming_init(self.conv_v_right, mode='fan_in')
        kaiming_init(self.conv_q_left, mode='fan_in')
        kaiming_init(self.conv_v_left, mode='fan_in')

        self.conv_q_right.inited = True
        self.conv_v_right.inited = True
        self.conv_q_left.inited = True
        self.conv_v_left.inited = True

    def spatial_pool(self, x):

        g_x = self.conv_q_left(x)
        batch, channel, height, width = g_x.size()
        avg_x = self.avg_pool(g_x)
        batch, channel, avg_x_h, avg_x_w = avg_x.size()
        avg_x = avg_x.view(batch, channel, avg_x_h * avg_x_w).permute(0, 2, 1)

        theta_x = self.conv_v_left(x).view(batch, self.inter_planes, height * width)
        context = torch.matmul(avg_x, theta_x)
        context = self.softmax_left(context)
        context = context.view(batch, 1, height, width)
        mask_sp = self.sigmoid(context)
        out = x * mask_sp
        return out

    def triplechannel_pool(self, x):

        g_x = self.conv_q_right(x)
        batch, channel, height, width = g_x.size()
        avg_x = self.avg_pool(g_x)
        batch, channel, avg_x_h, avg_x_w = avg_x.size()
        avg_x = avg_x.view(batch, channel, avg_x_h * avg_x_w).permute(0, 2, 1)

        theta_x = self.conv_v_right(x).view(batch, self.inter_a1, height * width)
        context = torch.matmul(avg_x, theta_x)
        context = self.softmax_left(context)
        context = context.view(batch, 1, height, width)
        mask_sp = self.sigmoid(context)
        out = x * mask_sp
        return out

    def forward(self, x):

        context_spatial = self.spatial_pool(x)
        x1 = x.permute(0,2,1,3).contiguous()
        context_channelH = self.triplechannel_pool(x1)
        context_channelH=context_channelH.permute(0,2,1,3).contiguous()

        x2 = x.permute(0, 3, 2, 1).contiguous()
        context_channelW = self.triplechannel_pool(x2)
        context_channelW = context_channelW.permute(0, 3, 2, 1).contiguous()
        out = context_spatial + context_channelH+context_channelW
        return out

class DPCTN(nn.Module):
    def __init__(self, channel=32):
        super().__init__()
         # ---- ResNet Backbone ----
        self.effit = EfficientNet.from_pretrained('efficientnet-b7')

        self.rfb2_1 = Conv(80, 32,3,1,padding=1,bn_acti=True)
        self.rfb3_1 = Conv(224, 32,3,1,padding=1,bn_acti=True)
        self.rfb4_1 = Conv(640, 32,3,1,padding=1,bn_acti=True)

        self.agg1 = HA(channel)
        self.edge = LA(in_fea=[32,48,80],mid_fea=32)
        self.trans=CCT(False, 48,channel_num=[80, 224, 640],patchSize=[4,2,1])
        self.ra1_conv1 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra1_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra1_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)

        self.ra2_conv1 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra2_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra2_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)

        self.ra3_conv1 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra3_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra3_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)

        self.aa_kernel_1 = TSA(32, 32, 48)
        self.aa_kernel_2 = TSA(32, 32, 24)
        self.aa_kernel_3 = TSA(32, 32, 12)

        self.pstrans = ps_vit_b_18()
        self.upde = dependencymap()

        self.grath1 = CGRModule(32,32,4)
        self.grath2 = CGRModule(32, 32, 4)
        self.grath3 = CGRModule(32, 32, 4)

        self.bifusion2 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.bifusion3 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.bifusion4 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.residucal2 = Residual(32*3,32)
        self.residucal3 = Residual(32 * 3, 32)
        self.residucal4 = Residual(32 * 3, 32)
        self.dp3=Conv(32,32,3,2,padding=1,bn_acti=True)
        self.dp4= Conv(32,32,3,2,padding=1,bn_acti=True)

        self.egdp1 = Conv(1, 1, 3, 2, padding=1, bn_acti=True)
        self.egde21 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.egde22 = Conv(32,1,3,1,padding=1,bn_acti=True)
        self.egde31 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.egde32 = Conv(32, 1, 3, 1, padding=1, bn_acti=True)
        self.egde41 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.egde42 = Conv(32, 1, 3, 1, padding=1, bn_acti=True)


    def forward(self, x):
        ##CNN
        endpoints = self.effit.extract_endpoints(x)
        x1 = endpoints['reduction_1']
        x2 = endpoints['reduction_2']
        x3 = endpoints['reduction_3']
        x4 = endpoints['reduction_4']
        x5 = endpoints['reduction_5']

        ##CCT
        x2h,x3h,x4h,_ =self.trans(x3,x4,x5)
        ##Transformer
        global_contexual = self.pstrans(x)
        x2t = self.upde(global_contexual)
        x3t= self.dp3(x2t)
        x4t = self.dp4(x3t)

        x2_rfb = self.rfb2_1(x2h)
        x3_rfb = self.rfb3_1(x3h)
        x4_rfb = self.rfb4_1(x4h)

        ##HA and LA
        de_1 = self.agg1(x4_rfb, x3_rfb, x2_rfb)
        egde_1 = self.edge(x1, x2, x2h)
        egde_1 = self.egdp1(egde_1)
        eqfusin1 = de_1.mul(egde_1)
        lateral_map_1 = F.interpolate(eqfusin1, scale_factor=8, mode='bilinear')  # #20 1 256 256

        ##DFC1
        x2p = self.aa_kernel_1(x2_rfb)
        x2_dp = self.bifusion2(x2_rfb*x2t)
        x2h = x2_rfb*x2p
        x2_all = self.residucal2(torch.cat([x2h,x2t,x2_dp],dim=1))
        ##DFC2
        x3p = self.aa_kernel_2(x3_rfb)
        x3_dp = self.bifusion3(x3_rfb * x3t)
        x3h = x3_rfb * x3p
        x3_all = self.residucal2(torch.cat([x3h, x3t, x3_dp], dim=1))
        ##DFC3
        x4p = self.aa_kernel_3(x4_rfb)
        x4_dp = self.bifusion4(x4_rfb * x4t)
        x4h = x4_rfb * x4p
        x4_all = self.residucal2(torch.cat([x4h, x4t, x4_dp], dim=1))
        #decoder1
        decoder_2 = F.interpolate(de_1, scale_factor=0.25, mode='bilinear')
        decoderegde_2 = F.interpolate(egde_1, scale_factor=0.25, mode='bilinear')
        decodereqfusin_2 = F.interpolate(eqfusin1, scale_factor=0.25, mode='bilinear')
        decoder_2_ra = -1 * (torch.sigmoid(decoder_2)) + 1
        aa_atten_3_o = decoder_2_ra.expand(-1, 32, -1, -1).mul(x4_all)
        ra_3 = self.ra3_conv1(aa_atten_3_o)
        ra_3 = self.ra3_conv2(ra_3)
        ra_3 = self.ra3_conv3(ra_3)
        egde_2 = self.grath1(x4_all,decoderegde_2)
        egde_2 = self.egde21(egde_2)
        egde_2 = self.egde22(egde_2)
        eqfusin2 =ra_3.mul(egde_2)
        lateral_map_21= eqfusin2+decodereqfusin_2
        lateral_map_2 = F.interpolate(lateral_map_21, scale_factor=32, mode='bilinear')
        #decoder2
        decoder_3 = F.interpolate(ra_3, scale_factor=2, mode='bilinear')
        decoderegde_3 = F.interpolate(egde_2, scale_factor=2, mode='bilinear')
        decodereqfusin_3 = F.interpolate(lateral_map_21, scale_factor=2, mode='bilinear')
        decoder_3_ra = -1 * (torch.sigmoid(decoder_3)) + 1
        aa_atten_2_o = decoder_3_ra.expand(-1, 32, -1, -1).mul(x3_all)
        ra_2 = self.ra2_conv1(aa_atten_2_o)
        ra_2 = self.ra2_conv2(ra_2)
        ra_2 = self.ra2_conv3(ra_2)
        egde_3 = self.grath2(x3_all, decoderegde_3)
        egde_3 =self.egde31(egde_3)
        egde_3 =self.egde32(egde_3)
        eqfusin3 = ra_2.mul(egde_3)
        lateral_map_31 = eqfusin3+decodereqfusin_3
        lateral_map_3 = F.interpolate(lateral_map_31, scale_factor=16, mode='bilinear')
        #decoder3
        decoder_4 = F.interpolate(ra_2, scale_factor=2, mode='bilinear')
        decoderegde_4 = F.interpolate(egde_3, scale_factor=2, mode='bilinear')
        decodereqfusin_4 = F.interpolate(lateral_map_31, scale_factor=2, mode='bilinear')
        decoder_4_ra = -1 * (torch.sigmoid(decoder_4)) + 1
        aa_atten_1_o = decoder_4_ra.expand(-1, 32, -1, -1).mul(x2_all)
        ra_1 = self.ra1_conv1(aa_atten_1_o)
        ra_1 = self.ra1_conv2(ra_1)
        ra_1 = self.ra1_conv3(ra_1)
        egde_4 = self.grath3(x2_all, decoderegde_4)
        egde_4 = self.egde41(egde_4)
        egde_4 = self.egde42(egde_4)
        eqfusin4 = ra_1.mul(egde_4)
        lateral_map_51 = eqfusin4+decodereqfusin_4
        lateral_map_5 = F.interpolate(lateral_map_51, scale_factor=8, mode='bilinear')
        lateral_map_egde5 = F.interpolate(egde_4, scale_factor=8, mode='bilinear')
        return lateral_map_5,lateral_map_3,lateral_map_2,lateral_map_1,lateral_map_egde5
