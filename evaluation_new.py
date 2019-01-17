import torch
from torchvision.utils import make_grid
from torchvision.utils import save_image

from util.image import unnormalize


def evaluate_new(model, dataset, device, filename):
    print("zip",type(zip(dataset)))
    image, mask, gt = zip(*[dataset[i] for i in range(1)])
    #image, mask, gt = zip(*[dataset[0]])
    filename_inpainting = './ws/inpaint/inpaiting.jpg'
    image = torch.stack(image)
    mask = torch.stack(mask)
    gt = torch.stack(gt)
    with torch.no_grad():
        output, _ = model(image.to(device), mask.to(device))
    output = output.to(torch.device('cpu'))
    output_comp = mask * image + (1 - mask) * output

    grid = make_grid(
        torch.cat((unnormalize(image), mask, unnormalize(output),
                   unnormalize(output_comp), unnormalize(gt)), dim=0))
    save_image(grid, filename)
    save_image(unnormalize(output_comp),filename_inpainting)
