import argparse
import os
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter
import torch.utils.data as data

from dataloader import Dataset, InfiniteSampler, img_transform
from network import net_train

def get_arguments():
    
    parser = argparse.ArgumentParser()
    
    #DATA DIRECTORIES
    parser.add_argument('--content_dir', type=str, default = "../DATA/content/train")#Train Content DIR
    parser.add_argument('--style_dir', type=str, default = "../DATA/style/train")#Train Style DIR
    parser.add_argument('--val_content_dir', type=str, default = "../DATA/content/validation")#Val Content DIR
    parser.add_argument('--val_style_dir', type=str, default = "../DATA/style/validation")#Val Style DIR
    #ENCODER 
    parser.add_argument('--vgg', type=str, default='./models/vgg_normalised.pth')#Pretrained Vgg
    #RESUME TRAINING
    parser.add_argument('--load_model', type=str, default = None)#Path to the resume model
    parser.add_argument('--start_iter', type=int, default = -1)#Iteration the resume model ended at
    #TRAINING PARAMETERS
    parser.add_argument('--lr', type=float, default=1e-4)#learning rate
    #parser.add_argument('--lr_decay', type=float, default=5e-5)
    parser.add_argument('--max_iter', type=int, default=100)#number of iterations to run for
    parser.add_argument('--batch_size', type=int, default=4)#batchsize
    parser.add_argument('--n_threads', type=int, default=0)#Argument for number of threads in dataloader
    parser.add_argument('--val_size', type = int, default= 10)#Number of (content, style) pairs in validation
    parser.add_argument('--val_batch_size', type = int, default= 2)#Batch size for validation
    #LOSS WEIGHTS
    parser.add_argument('--style_weight', type=float, default=7.0)#style weight
    parser.add_argument('--content_weight', type=float, default=1.0)#content weight
    #LOG DIRECTORIES
    parser.add_argument('--model_dir', default='./LOGS/model')#Dir to store the models
    parser.add_argument('--log_dir', default='./LOGS/tensorboard')#Dir to store the tensorboard files
    #LOG SAVE INTERVALS
    parser.add_argument('--save_train_log_interval', type = int, default = 10)#Save train loss every 'n' iterations
    parser.add_argument('--save_val_log_interval', type = int, default = 10)#Save val loss every 'n' iterations
    parser.add_argument('--save_model_interval', type=int, default=10000)#Save model weights every 'n' iterations
    
    args = parser.parse_args()
    return args


def main(args):
    
    #SET DEVICE
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('DEVICE SET TO: ', device.type, '...')
    
    #CHECK SAVING DIRECTORIES
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    print('ALL SAVE DIRECTORIES CHECKED...')
    
    #INTITIALISE SUMMARY WRITER
    writer = SummaryWriter(log_dir=args.log_dir)
    print('SUMMARY WRITER INITIALISED...')
    
    #LOAD THE DATASET
    tf = img_transform()
    train_dataset_content = Dataset(args.content_dir, tf)
    train_dataset_style = Dataset(args.style_dir, tf)
    
    val_dataset_content = Dataset(args.val_content_dir, tf, args.val_size)
    val_dataset_style = Dataset(args.val_style_dir, tf, args.val_size)
    
    train_iter_content = iter(data.DataLoader(train_dataset_content, 
                                      batch_size=args.batch_size,
                                      sampler=InfiniteSampler(len(train_dataset_content)),
                                      num_workers=args.n_threads))
    train_iter_style = iter(data.DataLoader(train_dataset_style, 
                                      batch_size=args.batch_size,
                                      sampler=InfiniteSampler(len(train_dataset_style)),
                                      num_workers=args.n_threads))
    print('TRAIN ITER LOADED...')
    val_iter_content = iter(data.DataLoader(val_dataset_content, 
                                      batch_size=args.val_batch_size,
                                      sampler=InfiniteSampler(len(val_dataset_content)),
                                      num_workers=0))
    val_iter_style = iter(data.DataLoader(val_dataset_style, 
                                      batch_size=args.val_batch_size,
                                      sampler=InfiniteSampler(len(val_dataset_style)),
                                      num_workers=0))
    print('VAL ITER LOADED...')
    
    
    #LOAD MODELS, LOSS, OPTIMISER
    model = net_train(args.vgg, args.load_model).to(device)
    start = args.start_iter + 1
    print('STARTING TRAINING AT ITERATION: ', start, '...')
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=args.lr)
    
    print('TRAIN AND VAL LOSS WILL BE STORED IN TENSORBOARD...')
    
    for i in tqdm(range(start, args.max_iter)):
        model.train()
        content_images = next(train_iter_content)
        style_images = next(train_iter_style)
        loss_c, loss_s = model(content_images.to(device), style_images.to(device), alpha = 1.0)
        loss_c = args.content_weight * loss_c
        loss_s = args.style_weight * loss_s
        loss = loss_c + loss_s

        #Store Train Log
        if i % args.save_train_log_interval == 0 or i == args.max_iter-1:
            writer.add_scalar('loss_content', loss_c.item(), i)
            writer.add_scalar('loss_style', loss_s.item(), i)
            writer.add_scalar('total_loss', loss.item(), i)
            writer.file_writer.flush()
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #Store Val Log
        if i % args.save_val_log_interval == 0:
            model.eval()
            val_n = int(args.val_size/args.val_batch_size)               
            val_total_loss_c= 0.0
            val_style_loss_c = 0.0
            val_content_loss_c = 0.0                
            for k in range(val_n):
                val_content_images = next(val_iter_content).to(device)
                val_style_images = next(val_iter_style).to(device)
                val_content_loss, val_style_loss = model(val_content_images, val_style_images, alpha = 1.0)  
                val_style_loss_c += val_style_loss.item()
                val_content_loss_c += val_content_loss.item()
                val_total_loss_c += (val_content_loss + val_style_loss).item()
                        
            writer.add_scalar('val_total_loss', val_total_loss_c/(val_n), i ) 
            writer.add_scalar('val_style_loss', val_style_loss_c/(val_n), i )
            writer.add_scalar('val_content_loss', val_content_loss_c/(val_n), i) 
            writer.file_writer.flush()
            
        #Save model
        if i % args.save_model_interval == 0 or i == args.max_iter -1:
            state_dict = model.decoder.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            torch.save(state_dict,
                       '{:s}/decoder_iter_{:d}.pth'.format(args.model_dir,i))
    
    
    
    
    writer.close()    
    
if __name__ == '__main__':
    args = get_arguments()
    main(args)