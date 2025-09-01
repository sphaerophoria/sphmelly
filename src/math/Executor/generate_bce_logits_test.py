import torch
import json
import sys
from torch.nn.functional import binary_cross_entropy_with_logits

def jsonTensor(t):
    return {
        "shape": list(reversed(t.shape)),
        "data": t.reshape(torch.numel(t)).tolist(),
    }

torch.manual_seed(0)


input = torch.randn(20, requires_grad=True)
input.retain_grad()

target = (torch.rand(20) < 0.5).to(torch.float32)

loss = binary_cross_entropy_with_logits(input, target, reduction='none')
loss.retain_grad()

(loss * torch.arange(20)).sum().backward()

print(json.dumps({
    "input": jsonTensor(input),
    "target": jsonTensor(target),
    "downstream_grad": jsonTensor(loss.grad),
    "input_grad": jsonTensor(input.grad),
    "output": jsonTensor(loss),
}));
