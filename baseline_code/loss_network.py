from vgg import Vgg16
import torch.nn as nn
import torch


def gram(x):
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w*h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (ch * h * w)
    return G

class CalculateLoss(nn.Module):
    def __init__(self, vgg_path):
        super().__init__()
        self.vgg = Vgg16(vgg_path)
        self.loss_mse = nn.MSELoss()
        self.style_weight = 1e5
        self.content_weight = 1e0
        self.tv_weight = 1e-7
    def add_style_img(self, style_img_tensor):
        self.style_features = self.vgg(style_img_tensor)
        self.style_gram = [gram(fmap) for fmap in self.style_features]
                

    def forward(self, content, output):
        
        batch_size_curr = int(content.shape[0])
        content_features = self.vgg(content)
        output_features = self.vgg(output)
        
        #STYLE LOSS
        output_gram = [gram(fmap) for fmap in output_features]
        style_loss = 0.0
        for k in range(4):
            style_loss += self.loss_mse(output_gram[k], self.style_gram[k][:batch_size_curr])
        style_loss = self.style_weight*style_loss
        
        #CONTENT LOSS======================================================
        recon = content_features[1]      
        recon_hat = output_features[1]
        content_loss = self.content_weight*self.loss_mse(recon_hat, recon)
        
        #TOTAL VARIATION LOSS==============================================
        diff_i = torch.sum(torch.abs(output[:, :, :, 1:] - output[:, :, :, :-1]))
        diff_j = torch.sum(torch.abs(output[:, :, 1:, :] - output[:, :, :-1, :]))
        tv_loss = self.tv_weight*(diff_i + diff_j)
        
        total_loss = style_loss + content_loss + tv_loss
            
        return total_loss, style_loss.item(), content_loss.item(), tv_loss.item()