import argparse
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid

from network import net_inference



def generate_image(tensor, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    """Save a given Tensor into an image file.
    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    return im

def test_transform():
    transform_list = []
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def stylize(content_path, style_path, alpha = 0.9):
    content = content_tf(Image.open(content_path))
    style = style_tf(Image.open(style_path))
    style = style.to(device).unsqueeze(0)
    content = content.to(device).unsqueeze(0)
    with torch.no_grad():
        output = model(content, style, alpha)
    output = output.cpu()
    output_img = generate_image(output)
    return output_img


#SET DEVICE
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

encoder_path = "./MODELS/vgg_normalised.pth"
decoder_path = "./MODELS/decoder_iter_90000.pth"
model = net_inference(encoder_path, decoder_path)
model.to(device)
model.eval()

content_tf = test_transform()
style_tf = test_transform()


if __name__ == '__main__':
    #Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_path', type=str, default = "./SAMPLE_TEST_IMAGES/content/bridge.jpg")#Content Image path
    parser.add_argument('--style_path', type=str, default = "./SAMPLE_TEST_IMAGES/style/scene_de_rue.jpg")#Style Image path
    parser.add_argument('--filename', type = str, default = "./RESULTS_inference/inference_1.jpg")
    args = parser.parse_args()
    img = stylize(args.content_path, args.style_path, alpha = 0.5)
    img.save(args.filename)



