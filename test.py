import sys
import os
import requests

import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import argparse

import models_mae

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def show_image(imaage, title=''):
	assert image.shape[2] == 3
	plt.imshow(torch.clip((image * imagenet_std + imagenet_mean)* 255,0,255).int())
	plt,title(title, fontsize=16)
	plt.axis('off')
	return

def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
	model = getattr(models_mae, arch)()
	checkpoint = torch.load(chkpt_dir, map_location='cpu')
	msg = model.load_state_dict(checkpoint['model'], strict=False)
	print(msg)
	return model

def run_one_image(img, model):
	x = torch.tensor(img)

	x = x.unsqueeze(dim=0)
	x = torch.einsum('nhwc->nchw',x)

	loss, y, mask = model(x.float(), mask_ratio=0.75)
	y = model.unpatchify(y)
	y = torch.einsum('nchw->nhwc',y).detach().cpu()

	mask = mask.detach()
	mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)
	mask = model.unpatchify(mask)
	mask = torch.einsum('nchw->nhwc',mask).detach().cpu()

	x = torch.einsum('nchw->nhwc', x)

	im_masked = x * (1 - mask)
	im_paste = x * (1 - mask) + y*mask

	plt.rcParams['figure.figsize'] = [24, 24]
	plt.subplot(1, 4, 1)
	show_image(x[0], "original")

	plt.subplot(1, 4, 2)
	show_image(im_masked[0], "masked")

	plt.subplot(1, 4, 3)
	show_image(y[0], "reconstruction")

	plt.subplot(1, 4, 4)
	show_image(im_paste[0], "reconstruction + visible")

	plt.savefig('test.png')
	#plt.show()
	plt.close()

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', metavar='img', type=str, help='Path to image')
    parser.add_argument('--chkpt', metavar='chkpt', type=str, default='./pretrained/mae_pretrain_vit_large.pth', help='Path to checkpoint')
    return parser.parse_args()

def main(args):

	img = Image.open(args.image)
	img = img.resize((224,224))
	img = np.array(img) / 255

	assert image.shape == (224, 224, 3)

	img = img - imagenet_mean
	img = img / imagenet_std

	#chkpt_dir = 'pretrained/mae_pretrain_vit_large.pth'
	model_mae = prepare_model(args.chkpt, 'mae_vit_large_patch16')
	print('Model loaded.')

	torch.manual_seed(2)
	run_one_image(img, model_mae)

if __name__ == '__main__':
    main(argparser())