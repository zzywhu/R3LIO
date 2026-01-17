
from REIN import REIN
import torch

model = REIN()
checkpoint = torch.load('/home/chx/BEVPlace/runs/Aug08_10-17-29/model_best.pth.tar', map_location=lambda storage, loc: storage)
if 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'])
else:
    model.load_state_dict(checkpoint)  
# to gpu
device = torch.device('cuda')
model.to(device)
model.eval()  

example_input = torch.rand(1, 3, 224, 224).to(device)

scripted_model = torch.jit.trace(model, example_input)

scripted_model.save("gpu.pt")
