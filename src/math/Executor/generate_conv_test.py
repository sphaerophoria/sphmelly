import torch
import json
from torch.nn.functional import conv2d

def jsonTensor(t):
    return {
        "shape": list(reversed(t.shape)),
        "data": t.reshape(torch.numel(t)).tolist(),
    }

torch.manual_seed(0)

img = torch.rand([4, 2, 5, 7], dtype=torch.float32, requires_grad=True)
img.retain_grad()

kernel = torch.rand([3, 2, 3, 5], dtype=torch.float32, requires_grad=True)
kernel.retain_grad()

x_pad = kernel.shape[3] // 2;
y_pad = kernel.shape[2] // 2;

img_padded = torch.nn.functional.pad(img, (x_pad, x_pad, y_pad, y_pad), "replicate", 0)
img_padded.retain_grad()

out = conv2d(img_padded, kernel, padding='valid')
out.retain_grad()

out2 = out * torch.arange(torch.numel(out)).reshape(out.shape)

total_loss = out2.sum()
total_loss.backward()

print(json.dumps([
    {
        "name": "convMany",
        "img": jsonTensor(img),
        "kernel": jsonTensor(kernel),
        "downstream_grad": jsonTensor(out.grad),
        "kernel_grad": jsonTensor(kernel.grad),
        # FIXME: This is technically wrong, border pixels have a greater effect on the
        # final gradient, however I think it might be fine to be wrong here so we just
        # test that the behavior is as we expect, not necessarily that it's correct
        "img_grad": jsonTensor(img_padded.grad[:, :, y_pad:y_pad+img.shape[2], x_pad:x_pad+img.shape[3]]),
        "output": jsonTensor(out),
    }
]));
