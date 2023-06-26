import torch
from bisenetv2 import BiSeNetV2

net=BiSeNetV2(2)
ckpt = torch.load('result/model/244--0.07386.pth', map_location="cpu")
net.eval()
net.load_state_dict(ckpt)
x = torch.randn((1, 3, 512, 1280))
traced_script_module = torch.jit.trace(net, x)
traced_script_module.save('deploy.pt')
