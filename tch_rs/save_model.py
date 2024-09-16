import torch
import torchvision

# 載入 ResNet 模型
model = torchvision.models.resnet18(pretrained=True)

# 編譯模型並存檔
model.eval()
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("model.pt")