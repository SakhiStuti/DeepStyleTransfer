import argparse
import os
import torch

from torch.optim import Adam
from torchvision import datasets
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from helper_functions import load_image, save_image
from torchvision import transforms
from loss_network import CalculateLoss
from net import style_network
import time



def get_arguments():
    parser = argparse.ArgumentParser(description='style transfer in pytorch')
    parser.add_argument('-mode', '--mode', type = str, default ='train', help = 'mode can either be train or test')
    #parser.add_argument('-gpu', '--gpu', type = int, default=0, help = 'set -1 for cpu / 0 or 1 etc according to the gpu')
    
    parser.add_argument('-train_dir', '--train_dir', type = str, default = '../baseline_DATA/content/train', help = 'directory containing content images')
    parser.add_argument('-sty', '--style_image_path', type = str, default = '../baseline_DATA//mosaic.jpg', help = 'path to the style image')
    parser.add_argument('-vgg_path', '--vgg_path', type = str, default = './vgg16-397923af.pth', help = 'path to the vgg model')
    parser.add_argument('-val_dir', '--val_dir', type = str, default = '../baseline_DATA/content/validation', help = 'directory containing content images for val')
    parser.add_argument('-test_img_dir', '--test_img_dir', type = str, default = '../baseline_DATA/content/test', help = 'directory containing content images for val')
    parser.add_argument('-load_model', '--load_model', type = str, default = None, help = 'model path for resume - /name.pth')
    
    parser.add_argument('-batch_size', '--batch_size', type = int, default = 4, help = 'batch size for training')
    parser.add_argument('-lr', '--lr', type = float, default = 1e-4, help = 'learning rate for learning')
    parser.add_argument('-epochs', '--epochs', type = int, default = 2, help = 'number of epochs to train')
    
    parser.add_argument("-log_dir", "--log_dir", type=str, default="./LOGS/tensorboard", help = 'dir saves the logs-tensorboard')
    parser.add_argument("-model_dir", "--model_dir", type=str, default="./LOGS/model", help = 'dir saves the trained models')
    parser.add_argument("-images_test_dir", "--images_test_dir", type = str, default = "./LOGS/test_image", help = 'dir saves the test result images')
    
    parser.add_argument("--log_interval", type=int, default=100, help = 'iterations after which train loss is stored')
    parser.add_argument("--model_interval", type=int, default=10000, help = 'iterations after which model is stored')
    parser.add_argument("--val_interval", type=int, default=500, help = 'iterations after which validation loss is stored')
    parser.add_argument("--test_image_interval", type = int, default = 500, help = 'iterations after which validation image results are stored') 
    
    return parser.parse_args()


def train(args):
    ##ALL REQUIRED INITIALIZATIONS
    
    #checking all saving directories
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
        
    if not os.path.exists(args.images_test_dir):
        os.makedirs(args.images_test_dir)
        
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    print('Saving paths checked...')
    
    #Summary Writer
    writer = SummaryWriter(args.log_dir)
    print('Summary Writer Initialised')
    
    #SET DEVICE
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('DEVICE SET TO: ', device.type, '...')
        
    loss = CalculateLoss(args.vgg_path).to(device) #send to gpu
    print('Loss function loaded...')
    
    style_net = style_network().to(device)
    print('Style Network loaded...')
    
    optimizer = Adam(style_net.parameters(), args.lr) 
    print('Optimizer set...')
    
    #Initialising style image tensor
    style_transform = transforms.Compose([
        transforms.ToTensor(), # turn image from [0-255] to [0-1] and convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # normalize with ImageNet values
    ])
    style_img = load_image(args.style_image_path) #Style image is of the size 256 x 256
    style_img = style_transform(style_img)
    style_img = style_img.repeat(args.batch_size, 1, 1, 1).to(device)
    
    loss.add_style_img(style_img)
    
    # Resume training on model
    start = 0
    if args.load_model:
        filename = args.model_dir + args.load_model
        checkpoint_dict = torch.load(filename)
        style_net.load_state_dict(checkpoint_dict["model"])
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
        start = checkpoint_dict["epoch"] + 1
        print("Resuming training on model:{} and epoch:{}".format(args.load_model, start))
        
        # Load all parameters to gpu
        style_net = style_net.to(device)
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
    
    #content images
    content_transform = transforms.Compose([
        transforms.Scale(256),           # scale shortest side to image_size
        transforms.CenterCrop(256),      # crop center image_size out
        transforms.ToTensor(),                  # turn image from [0-255] to [0-1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])      # normalize with ImageNet values
    ])
    
    #data loader for content images (train + val)
    train_dataset = datasets.ImageFolder(args.train_dir, content_transform)
    val_dataset = datasets.ImageFolder(args.val_dir, content_transform)
    
    
    #Load testing images
    test_images = []
    images = ['lighthouse.jpg', 'pier.jpg', 'sfbridge.jpg', 'skyline.jpg']
    for name in images:
        testImage = load_image(args.test_img_dir + '/' + name)
        testImage = content_transform(testImage)
        test_images.append(testImage.repeat(1, 1, 1, 1))
    
    b = float(args.batch_size)
    t_s = time.time()
    for i in range(start, start+args.epochs):
        content_loader = DataLoader(train_dataset, batch_size =args.batch_size, drop_last=True, shuffle=True)
        N = len(content_loader)
        #print(N)
        #train over whole dataset in 1 epoch
        for j, batch in enumerate(content_loader):
        
            batch_train_img = batch[0].to(device)
            
            output = style_net(batch_train_img)
            #print('output extracted')
            
            #zero out gradients
            optimizer.zero_grad()
            
            total_loss, style_loss_i, content_loss_i, tv_loss_i = loss(batch_train_img, output)
            total_loss_i = total_loss.item()
            
            
            #backprop
            total_loss.backward()
            optimizer.step()
            
            #Save train loss
            if(j) % args.log_interval == 0:
                
                writer.add_scalar('train_total_loss', total_loss_i/b, (i*N + j))
                writer.add_scalar('train_style_loss', style_loss_i/b, (i*N + j))
                writer.add_scalar('train_content_loss', content_loss_i/b, (i*N + j))
                writer.add_scalar('train_tv_loss', tv_loss_i/b, (i*N + j))
                writer.file_writer.flush()
                #print('Saved train loss...') 
            print(total_loss_i/b)
            
            #Save val image
            if(j) % args.val_interval == 0:
                style_net.eval()
                val_loader = DataLoader(val_dataset, batch_size = args.batch_size, drop_last=True)
                val_n = len(val_loader)                
                val_total_loss_c= 0.0
                val_style_loss_c = 0.0
                val_content_loss_c = 0.0
                val_tv_loss_c = 0.0	                
                for k, batch_val in enumerate(val_loader):
                    batch_val_img = batch[0].to(device)
                    output_val = style_net(batch_val_img)
                    val_total_loss, val_style_loss_i, val_content_loss_i, val_tv_loss_i = loss(batch_val_img, output_val)
                    val_total_loss_c += val_total_loss.item()
                    val_style_loss_c += val_style_loss_i
                    val_content_loss_c += val_content_loss_i
                    val_tv_loss_c += val_tv_loss_i
                    
                writer.add_scalar('val_total_loss', val_total_loss_c/(val_n*b), (i*N + j))  
                writer.add_scalar('val_style_loss', val_style_loss_c/(val_n*b), (i*N + j)) 
                writer.add_scalar('val_content_loss', val_content_loss_c/(val_n*b), (i*N + j)) 
                writer.add_scalar('val_tv_loss', val_tv_loss_c/(val_n*b), (i*N + j)) 
                style_net.train()
                #print('Saved val loss...') 
                print(val_total_loss_c/(val_n*b)) 
                
            #Save test image
            if(j) % args.test_image_interval == 0 or (j) == N-1:
                style_net.eval()
                k = 0
                for img in test_images:
                    outputTestImage = style_net(img.to(device)).cpu()
                    path = args.images_test_dir + ("/test_k{}_e{}_i{}.jpg".format(k, i, j))
                    save_image(path, outputTestImage.data[0])
                    k =k + 1
                style_net.train()
                
            #Save model
            # Save model
            if (j) % args.model_interval == 0 or (j) == N - 1:
                filename = args.model_dir + "/model_e{}_i{}.pth".format(i, j)
                state = {"model": style_net.state_dict(), "optimizer": optimizer.state_dict(), "epoch": i}
                torch.save(state, filename)
                #print('Saved model')  
          
        
    writer.close()  
    t_e = time.time()
    print(t_e - t_s)          
                
            
            
def main(args):
    
    if args.mode == 'train':
        train(args)


if __name__ == '__main__':
    args = get_arguments()
    print('Arguments loaded')
    main(args)
