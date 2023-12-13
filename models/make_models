import os
import torch

model1 = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="celeba_distill")
model2 = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="face_paint_512_v1")
model3 = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="face_paint_512_v2")
model4 = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="paprika")

model1.cpu()
model2.cpu()
model3.cpu()
model4.cpu()
model1.eval()
model2.eval()
model3.eval()
model4.eval()

traced_script_module1 = torch.jit.trace(model1, torch.randn(1, 3, 512, 512), strict=False)
traced_script_module1.save("celeba_distill.pt")

traced_script_module2 = torch.jit.trace(model2, torch.randn(1, 3, 512, 512), strict=False)
traced_script_module2.save("face_paint_512_v1.pt")

traced_script_module3 = torch.jit.trace(model3, torch.randn(1, 3, 512, 512), strict=False)
traced_script_module3.save("face_paint_512_v2.pt")

traced_script_module4 = torch.jit.trace(model4, torch.randn(1, 3, 512, 512), strict=False)
traced_script_module4.save("paprika.pt")

os.system("pnnx celeba_distill.pt inputshape=[1,3,512,512] device=cpu") 
os.system("pnnx face_paint_512_v1.pt inputshape=[1,3,512,512] device=cpu") 
os.system("pnnx face_paint_512_v2.pt inputshape=[1,3,512,512] device=cpu") 
os.system("pnnx paprika.pt inputshape=[1,3,512,512] device=cpu") 


