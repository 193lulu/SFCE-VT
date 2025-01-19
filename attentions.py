import numpy as np
import random
import torch
import torch.nn.functional as F
import time


def attention_crop_drop(attention_maps, input_image):
    B,N,W,H = input_image.shape
    input_tensor = input_image
    attention_maps1 = torch.tensor([item.cpu().detach().numpy() for item in attention_maps]).cuda()
    batch_size, num_parts,bast_batch, height, width = attention_maps1.shape
    attention_maps1 = attention_maps1.view(batch_size * num_parts,bast_batch, height, width)
    attention_maps = torch.nn.functional.interpolate(attention_maps1.detach(),size=(W,H),mode='bilinear')
    part_weights = F.avg_pool2d(attention_maps1.detach(),(W,H)).reshape(batch_size,-1)
    part_weights = torch.add(torch.sqrt(part_weights),1e-12)
    part_weights = torch.div(part_weights,torch.sum(part_weights,dim=1).unsqueeze(1)).cpu()
    part_weights = part_weights.numpy()
    ret_imgs = []
    masks = []
    for i in range(batch_size):
        attention_map = attention_maps[i]
        part_weight = part_weights[i]
        selected_index = np.random.choice(np.arange(0, len(part_weight)), 1, p=part_weight)[0]
        selected_index2 = np.random.choice(np.arange(0, len(part_weight)), 1, p=part_weight)[0]
        mask = attention_map[selected_index, :]
        threshold = random.uniform(0.4, 0.6)#随机阈值
        itemindex = torch.nonzero(mask >= threshold*mask.max())
        padding_h = int(0.1*H)
        padding_w = int(0.1*W)
        height_min = itemindex[:,0].min()
        height_min = max(0,height_min-padding_h)
        height_max = itemindex[:,0].max() + padding_h
        width_min = itemindex[:,1].min()
        width_min = max(0,width_min-padding_w)
        width_max = itemindex[:,1].max() + padding_w
        out_img = input_tensor[i][:,height_min:height_max,width_min:width_max].unsqueeze(0)
        out_img = torch.nn.functional.interpolate(out_img,size=(W,H),mode='bilinear',align_corners=True)
        out_img = out_img.squeeze(0)
        ret_imgs.append(out_img)
        mask2 = attention_map[selected_index2:selected_index2 + 1, :, :]
        mask2 = (mask2 > threshold * mask2.max()).float()
        masks.append(mask2)
    crop_imgs = torch.stack(ret_imgs)
    masks = torch.stack(masks)
    drop_imgs = input_tensor*masks
    return (crop_imgs,drop_imgs)
