from PIL import Image
from torchvision import transforms
import os
import numpy as np

def load_image(filename):
    img = Image.open(filename)
    return img

def gram(x):
    #Calculates the gram matrix
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w*h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (ch * h * w)
    return G

def check_dir_train(args):
    
    #add log directory if does not exist
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
        
    if not os.path.exists(args.images_val_dir):
        os.makedirs(args.images_val_dir)
        
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
        
    print('Saving paths checked')


def save_image(filename, data):
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    img = data.clone().numpy()
    img = ((img * std + mean).transpose(1, 2, 0)*255.0).clip(0, 255).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)
