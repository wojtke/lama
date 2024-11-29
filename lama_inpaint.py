import torch
#from saicinpainting.training.modules.ffc import FFCResNetGenerator as FFCResNetGenerator_OG
from ffc2 import FFCResNetGenerator as FFCResNetGenerator_NEW
import torch.nn.functional as F
import lovely_tensors
lovely_tensors.monkey_patch()
# from debug import print_flow

torch.set_grad_enabled(False)

# CONFIG = {
#     'input_nc': 4, 'output_nc': 3, 'ngf': 64, 'n_downsampling': 3, 'n_blocks': 18, 'add_out_act': 'sigmoid', 
#     'init_conv_kwargs': {'ratio_gin': 0, 'ratio_gout': 0, 'enable_lfu': False}, 
#     'downsample_conv_kwargs': {'ratio_gin': r'${generator.init_conv_kwargs.ratio_gout}', 'ratio_gout': r'${generator.downsample_conv_kwargs.ratio_gin}', 'enable_lfu': False}, 
#     'resnet_conv_kwargs': {'ratio_gin': 0.75, 'ratio_gout': r'${generator.resnet_conv_kwargs.ratio_gin}', 'enable_lfu': False}
# }

CONFIG = {
    'input_nc': 4, 'output_nc': 3, 'ngf': 64, 'n_downsampling': 3, 'n_blocks': 18, 'add_out_act': 'sigmoid', 
    'init_conv_kwargs': {'ratio_gin': 0, 'ratio_gout': 0, 'enable_lfu': False}, 
    'downsample_conv_kwargs': {'ratio_gin': 0.75, 'ratio_gout': 0.75, 'enable_lfu': False}, 
    'resnet_conv_kwargs': {'ratio_gin': 0.75, 'ratio_gout': 0.75, 'enable_lfu': False}
}



class Dilate(torch.nn.Module):
    def __init__(self, scale=3):
        super(Dilate, self).__init__()
        self.scale = scale
        self.max_pool = torch.nn.MaxPool2d(kernel_size=scale, padding=0)

    def forward(self, x):
        _, _, h, w = x.size()
        x = self.max_pool(x)
        x = F.interpolate(x, size=(h, w), mode="bilinear")
        return torch.ceil(x).clamp(0,1)



class LamaWrapper(torch.nn.Module):
    def __init__(self, OG=False) -> None:
        super().__init__()
        self.dilate = Dilate()
        #model = FFCResNetGenerator_OG(**CONFIG) if OG else FFCResNetGenerator_NEW(**CONFIG)

        model = FFCResNetGenerator_NEW()
        model.load_state_dict(torch.load("/Users/woj/lama/lama_state_dict.pt", map_location='cpu'), strict=False)
        model.eval()

        self.model = model

    def forward(self, img, mask):
        mask = self.dilate(mask)
        inp = torch.cat([img * (1-mask), mask], dim=1)
        result = self.model(inp)
        return result


if __name__ == "__main__":
    print("INIT")
    print("---------------------------")

    lama_og = LamaWrapper(OG=True)
    lama_new = LamaWrapper(OG=False)

    print("PRINT")
    print("---------------------------")
    print(lama_og)
    print("---------------------------")
    print(lama_new)


    print("FORWARD")
    print("---------------------------")
    example_inputs = (
        torch.rand(1, 3, 512, 512),
        torch.rand(1, 1, 512, 512)
    )

    res_og = lama_og(*example_inputs)
    res_new = lama_new(*example_inputs)

    print("result OG:", res_og)
    print("result NEW:", res_new)
    diff = res_og-res_new
    print("diff mean:", diff.mean())
    print("diff l2:", diff.square().sum().sqrt())

    # print("FLOW")
    # print("---------------------------")

    # #print_flow(lama, example_inputs)

