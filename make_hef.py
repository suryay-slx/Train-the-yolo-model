import torch
from torchvision import models
from hailo_sdk_client import ClientRunner

# Exporting the model to ONNX
device = torch.device("cuda")
model = models.resnet18(pretrained=False)
model_name = "resnet18"

batchSize = 8
inputs = torch.rand([batchSize, 3, 224, 224]).cuda()
model = model.to(device)
model.eval()

onnx_path = f'{model_name}.onnx'

torch.onnx.export(model, inputs, onnx_path,
export_params=True,
do_constant_folding=False,
)

# Exporting the ONNX to HAR(Hailo Archive Representation)
chosen_hw_arch = "hailo8l"
onnx_path = f"./{model_name}.onnx"

runner = ClientRunner(hw_arch=chosen_hw_arch)
hn, npz = runner.translate_onnx_model(
		onnx_path,
		onnx_model_name,
)
hailo_model_har_name = f"{model_name}.har"
runner.save_har(hailo_model_har_name)

#Compiling HAR to HER(Hailo Executable File)
hailo_model_har_name = f"{model_name}.har"
runner = ClientRunner(har=hailo_model_har_name)
hef = runner.compile()

file_name = f"{model_name}.hef"
with open(file_name, ”wb”) as f:
    f.write(hef)

