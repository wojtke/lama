import torch
import torch.nn as nn


class FourierUnit(nn.Module):

    def __init__(self):
        super(FourierUnit, self).__init__()
        self.conv_layer = torch.nn.Conv2d(384, 384, kernel_size=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(384)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        batch = x.shape[0]
        fft_dim = (-2, -1)
        #print(x)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm='ortho')
        ffted = torch.view_as_real(ffted).permute(0, 1, 4, 2, 3).contiguous() # (batch, c, 2, h, w/2+1)
        #print(ffted)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        #print(ffted)
        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        #print(ffted)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])
        #print(ffted.size())
        ifft_shape_slice = x.shape[-2:]
        # print("ifft_shape_slice", ifft_shape_slice)
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm='ortho')
        #print(output)
        #print("----------------------")

        return output


class SpectralTransform(nn.Module):

    def __init__(self,):
        super(SpectralTransform, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(384, 192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit()
        self.conv2 = torch.nn.Conv2d(192, 384, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        output = self.fu(x)
        output = self.conv2(x + output)
        return output


class ConcatTupleLayer(nn.Module):
    def forward(self, x):
        return torch.cat(x, dim=1)
    

class FFC_1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super().__init__()
        self.convl2l = nn.Conv2d(in_channels, out_channels, bias=False, 
                                 stride=stride, kernel_size=kernel_size, 
                                 padding=padding, padding_mode="reflect")

    def forward(self, x):
        return self.convl2l(x)


class FFC_BN_ACT_1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super().__init__()
        self.ffc = FFC_1(in_channels, out_channels, kernel_size, padding, stride)
        self.bn_l = nn.BatchNorm2d(out_channels)
        self.act_l = nn.ReLU(inplace=True) 

    def forward(self, x):
        x = self.ffc(x)
        x = self.act_l(self.bn_l(x))
        return x
    

class FFC_2(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2, kernel_size=3, padding=0, stride=1):
        super().__init__()
        self.convl2l = nn.Conv2d(in_channels, out_channels1, bias=False, 
                                 stride=stride, kernel_size=kernel_size, 
                                 padding=padding, padding_mode="reflect")
        
        self.convl2g = nn.Conv2d(in_channels, out_channels2, bias=False, 
                                 stride=stride, kernel_size=kernel_size, 
                                 padding=padding, padding_mode="reflect")


    def forward(self, x):
        return self.convl2l(x), self.convl2g(x)


class FFC_BN_ACT_2(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2, kernel_size, padding=0, stride=1):
        super().__init__()
        self.ffc = FFC_2(in_channels, out_channels1, out_channels2, kernel_size, padding, stride)
        self.bn_l = nn.BatchNorm2d(out_channels1)
        self.bn_g = nn.BatchNorm2d(out_channels2)
        self.act_l = nn.ReLU(inplace=True) 
        self.act_g = nn.ReLU(inplace=True) 

    def forward(self, x):
        x1, x2 = self.ffc(x)
        x1 = self.act_l(self.bn_l(x1))
        x2 = self.act_g(self.bn_g(x2))
        return x1, x2
    

#######
    

class FFC_3(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_size=3
        padding=1
        stride=1
        self.convl2l = nn.Conv2d(128, 128, bias=False, 
                                 stride=stride, kernel_size=kernel_size, 
                                 padding=padding, padding_mode="reflect")
        
        self.convl2g = nn.Conv2d(128, 384, bias=False, 
                                 stride=stride, kernel_size=kernel_size, 
                                 padding=padding, padding_mode="reflect")
        self.convg2l = nn.Conv2d(384, 128, bias=False, 
                                 stride=stride, kernel_size=kernel_size, 
                                 padding=padding, padding_mode="reflect")
        self.convg2g = SpectralTransform()
        

    def forward(self, x):
        x_l, x_g = x

        out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        out_xg = self.convl2g(x_l) + self.convg2g(x_g)
        return out_xl, out_xg

class FFC_BN_ACT_3(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffc = FFC_3()
        self.bn_l = nn.BatchNorm2d(128)
        self.bn_g = nn.BatchNorm2d(384)
        self.act_l = nn.ReLU(inplace=True) 
        self.act_g = nn.ReLU(inplace=True) 

    def forward(self, x):
        x1, x2 = self.ffc(x)
        x1 = self.act_l(self.bn_l(x1))
        x2 = self.act_g(self.bn_g(x2))
        return x1, x2
    

class FFCResnetBlock_clean(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = FFC_BN_ACT_3()
        self.conv2 = FFC_BN_ACT_3()

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)

        id_l, id_g = x_l, x_g

        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))

        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g
        return out
        
    

class FFCResNetGenerator(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        model =[ 
            nn.ReflectionPad2d(3),                                                      # 0 
            FFC_BN_ACT_1(4, 64, kernel_size=7),                                         # 1 
            FFC_BN_ACT_1(64, 128, kernel_size=3, padding=1, stride=2),                  # 2 
            FFC_BN_ACT_1(128, 256, kernel_size=3, padding=1, stride=2),                 # 3 
            FFC_BN_ACT_2(256, 128, 384, kernel_size=3, padding=1, stride=2),            # 4                 
        ]

        for _ in range(18):                                                             #5-22
            model.append(
                FFCResnetBlock_clean()
            )


        model += [ 
            ConcatTupleLayer(),                                                                     # 23        
            nn.ConvTranspose2d(512, 256, kernel_size=3,  stride=2, padding=1, output_padding=1),    # 24 
            nn.BatchNorm2d(256),                                                                    # 25   
            nn.ReLU(inplace=True),                                                                  # 26         
            nn.ConvTranspose2d(256, 128, kernel_size=3,  stride=2, padding=1, output_padding=1),    # 27 
            nn.BatchNorm2d(128),                                                                    # 28   
            nn.ReLU(inplace=True),                                                                  # 29          
            nn.ConvTranspose2d(128, 64, kernel_size=3,  stride=2, padding=1, output_padding=1),     # 30 
            nn.BatchNorm2d(64),                                                                     # 31   
            nn.ReLU(inplace=True),                                                                  # 32
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, kernel_size=7),
            nn.Sigmoid()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

