import torch
import torch.nn as nn
import torchvision
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import numpy as np
from PIL import Image
import torch.nn.functional as F
#from logger import Logger
import math
def get_unnormalized_ds_item(unnormalize, item):
    un_img = unnormalize(item[0][None])
    return (un_img.squeeze(dim=0), item[1])

def freeze_all_layers(model):
    for param in model.parameters():
        param.requires_grad = False
        
def unfreeze_all_layers(model):
    for param in model.parameters():
        param.requires_grad = True        

def get_model_predictions_on_a_sample_batch(model, dl):
    model.eval()
    with torch.no_grad():
        batch, actual_labels = iter(dl).next()
        batch = batch.to(device)
        actual_labels = actual_labels.to(device)
        predictions = model(batch)
    
    return (predictions, batch, actual_labels)


def im_gradient_loss(d_batch, n_pixels):
    a = torch.Tensor([[[[1, 0, -1],
                      [2, 0, -2],
                      [1, 0, -1]]]])
                      
    b = torch.Tensor([[[[1, 2, 1],
                      [0, 0, 0],
                      [-1, -2, -1]]]])
    a = a.to(device)
    b = b.to(device)
    
    G_x = F.conv2d(d_batch, a, padding=1).to(device)
    G_y = F.conv2d(d_batch, b, padding=1).to(device)
    
    G = torch.pow(G_x,2)+ torch.pow(G_y,2)
    
    return G.view(-1, n_pixels).mean(dim=1).sum()

def depth_loss(preds, actual_depth):
    #preds.shape        -> [16, 1, 120, 160]
    #actual_depth.shape -> [16, 120, 160]
    n_pixels = actual_depth.shape[1]*actual_depth.shape[2]
    
    preds = (preds*0.225) + 0.45
    preds = preds*255
    preds[preds<=0] = 0.00001
    actual_depth[actual_depth==0] = 0.00001
    actual_depth.unsqueeze_(dim=1)
    d = torch.log(preds) - torch.log(actual_depth)

    grad_loss_term = im_gradient_loss(d, n_pixels)
    term_1 = torch.pow(d.view(-1, n_pixels),2).mean(dim=1).sum() #pixel wise mean, then batch sum
    term_2 = (torch.pow(d.view(-1, n_pixels).sum(dim=1),2)/(2*(n_pixels**2))).sum()
    
    return term_1 - term_2 + grad_loss_term
def err_rms_linear(preds, actual_depth):
    # preds.shape        -> [batch_size, 1, 120, 160]
    # actual_depth.shape -> [batch_size, 120, 160]
    n_pixels = actual_depth.shape[1] * actual_depth.shape[2] # 120*160

    # ac_min = actual_depth.view(8,-1).min(dim=1)[0]
    # ac_max = actual_depth.view(8,-1).max(dim=1)[0]
    # pr_min = preds.view(8,-1).min(dim=1)[0]
    # pr_max = preds.view(8,-1).max(dim=1)[0]
    # print(f"ac_min:{ac_min}\nac_max:{ac_max}\npr_min:{pr_min}\npr_max:{pr_max}")

    # 아래와 같이 loss가 설정되었으므로 아래를 따라야 한다.
    # preds = (preds * 0.225) + 0.45
    # preds = preds * 255
    pr_min = preds.view(8, -1).min(dim=1)[0]
    pr_min = pr_min.view(8,1,1,1)
    preds = preds - pr_min
    pr_max = preds.view(8, -1).max(dim=1)[0]
    pr_max = pr_min.view(8,1,1,1)
    preds = (preds/pr_max)*255
    preds = preds.view(8,120,160)
    #
    preds[preds <= 0] = 0.00001
    actual_depth[actual_depth == 0] = 0.00001
    # actual_depth.unsqueeze_(dim=1) # actual_depth 를
    #
    ans = torch.norm(actual_depth - preds) / math.sqrt(8*120*160)

    # diff = abs(preds - actual_depth)
    # print(f"00@@@@@@@@@@@@@@@@{diff.shape}")
    # diff_pow = torch.pow(diff, 2)
    # a = torch.sum(diff_pow, 1)
    # print(f"1@@@@@@@@@@@@@@@@{a.shape}")
    # a2 = torch.sum(a, 1)
    # print(f"2@@@@@@@@@@@@@@@@{a2.shape}")
    # a3 = a2/n_pixels
    # a4 = torch.sqrt(a3)
    # a5=a4.sum()
    return ans

def err_rms_log(preds, actual_depth):
    # preds.shape        -> [batch_size, 1, 120, 160]
    # actual_depth.shape -> [batch_size, 120, 160]
    n_pixels = actual_depth.shape[1] * actual_depth.shape[2] # 120*160

    # 아래와 같이 loss가 설정되었으므로 아래를 따라야 한다.
    preds = (preds * 0.225) + 0.45
    preds = preds * 255
    preds[preds <= 0] = 0.00001
    actual_depth[actual_depth == 0] = 0.00001
    actual_depth.unsqueeze_(dim=1) # actual_depth 를  -> [batch_size, 1, 120, 160]


    diff = torch.log(preds) - torch.log(actual_depth)
    diff_pow = torch.pow(diff, 2)
    a = torch.sum(diff_pow, 2)
    a2 = torch.sum(a, 2)
    a3 = a2/n_pixels
    a4 = torch.sqrt(a3)
    return a4.sum()

def err_abs_rel(preds, actual_depth):
    # preds.shape        -> [batch_size, 1, 120, 160]
    # actual_depth.shape -> [batch_size, 120, 160]
    n_pixels = actual_depth.shape[1] * actual_depth.shape[2] # 120*160

    # 아래와 같이 loss가 설정되었으므로 아래를 따라야 한다.
    preds = (preds * 0.225) + 0.45
    preds = preds * 255
    preds[preds <= 0] = 0.00001
    actual_depth[actual_depth == 0] = 0.00001
    actual_depth.unsqueeze_(dim=1) # actual_depth 를  -> [batch_size, 1, 120, 160]


    diff = abs(preds - actual_depth)
    diff = diff/actual_depth

    a = torch.sum(diff, 2)
    a2 = torch.sum(a, 2)
    a3 = a2/n_pixels

    return a3.sum()

def err_sql_rel(preds, actual_depth):
    # preds.shape        -> [batch_size, 1, 120, 160]
    # actual_depth.shape -> [batch_size, 120, 160]
    n_pixels = actual_depth.shape[1] * actual_depth.shape[2] # 120*160

    # 아래와 같이 loss가 설정되었으므로 아래를 따라야 한다.
    preds = (preds * 0.225) + 0.45
    preds = preds * 255
    preds[preds <= 0] = 0.00001
    actual_depth[actual_depth == 0] = 0.00001
    actual_depth.unsqueeze_(dim=1) # actual_depth 를  -> [batch_size, 1, 120, 160]

    diff = abs(preds - actual_depth)
    diff_pow = torch.pow(diff, 2)
    diff = diff_pow/actual_depth
    a = torch.sum(diff, 2)
    a2 = torch.sum(a, 2)
    a3 = a2/n_pixels
    a4=a3.sum()
    return a4

def err_psnr(preds, actual_depth):
    # preds.shape        -> [batch_size, 1, 120, 160]
    # actual_depth.shape -> [batch_size, 120, 160]
    n_pixels = actual_depth.shape[1] * actual_depth.shape[2] # 120*160
    # 아래와 같이 loss가 설정되었으므로 아래를 따라야 한다.
    preds = (preds * 0.225) + 0.45
    preds = preds * 255
    preds[preds <= 0] = 0.00001
    actual_depth[actual_depth == 0] = 0.00001
    actual_depth.unsqueeze_(dim=1) # actual_depth 를  -> [batch_size, 1, 120, 160]

    diff = abs(preds - actual_depth)
    diff_pow = torch.pow(diff, 2)
    a = torch.sum(diff_pow, 2)
    a2 = torch.sum(a, 2)
    a3 = a2 / n_pixels
    a3 = n_pixels/a3
    a4 = 10*torch.log10(a3)
    return a4.sum()


def print_training_loss_summary(loss, total_steps, current_epoch, n_epochs, n_batches, print_every=10):
    #prints loss at the start of the epoch, then every 10(print_every) steps taken by the optimizer
    steps_this_epoch = (total_steps%n_batches)
    
    if(steps_this_epoch==1 or steps_this_epoch%print_every==0):
        print ('Epoch [{}/{}], Iteration [{}/{}], Loss: {:.4f}' 
               .format(current_epoch, n_epochs, steps_this_epoch, n_batches, loss))


def apply_sobel_operator_on_sample_ds_image(ds_item, unnormalize, T, P):
    x = unnormalize(ds_item[0][None]) #send x as a batch of 1 item
    x_bw = P(x[0]).convert('L')
    x = T(x_bw)[None]

    #Black and white input image x, 1x1xHxW
    a = torch.Tensor([[1, 0, -1],
                      [2, 0, -2],
                      [1, 0, -1]])

    a = a.view((1,1,3,3))
    G_x = F.conv2d(x, a, padding=1)

    b = torch.Tensor([[1, 2, 1],
                      [0, 0, 0],
                      [-1, -2, -1]])

    b = b.view((1,1,3,3))
    G_y = F.conv2d(x, b, padding=1)

    G = torch.sqrt(torch.pow(G_x,2)+ torch.pow(G_y,2))
    return G_x[0], G_y[0], G[0]
