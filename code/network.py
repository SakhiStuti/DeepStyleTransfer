import torch
import torch.nn as nn
from encoder import encoder
from decoder import decoder
from helper_function import adaptive_instance_normalization as adain
from helper_function import calc_mean_std



    
class net_inference(nn.Module):
    """
    Returns the output Image
    """
    def __init__(self, encoder_path, decoder_path):
        super(net_inference, self).__init__()
        self.encoder = encoder(encoder_path)
        self.decoder = decoder(decoder_path)
        
        
    def forward(self, content, style, alpha):
        content_feat = self.encoder(content)
        style_feat = self.encoder(style)
        feat = adain(content_feat, style_feat)
        feat = alpha * feat + (1 - alpha) * content_feat
        output = self.decoder(feat)
        return output

  
class net_train(nn.Module):
    def __init__(self, encoder_path, decoder_path = None):
        super(net_train, self).__init__()
        self.encoder = encoder(encoder_path)
        self.decoder = decoder()
        self.decoder.load_state_dict(torch.load(decoder_path)) #Need to change this to be done in encoder
        self.mse_loss = nn.MSELoss()
    
    def calc_content_loss(self, input, target):
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)
               
    def forward(self, content, style, alpha):
        style_feats = self.encoder.get_intermediate_features(style)
        content_feat = self.encoder(content)
        t = adain(content_feat, style_feats[-1])
        t = alpha * t + (1 - alpha) * content_feat

        g_t = self.decoder(t)
        g_t_feats = self.encoder.get_intermediate_features(g_t)

        loss_c = self.calc_content_loss(g_t_feats[-1], t)
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        return loss_c, loss_s
        
        
        
    
