import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
from torch.autograd import grad
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import transforms
from model import Generator
from model import Segmentor
from PIL import Image


class Translater(object):
    
    def __init__(self, config):
        
        # Model hyper-parameters
        self.c_dim = config.c_dim
        self.s_dim = config.s_dim
        self.config = config
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.test_model = config.test_model

        self.G = Generator(self.g_conv_dim, self.c_dim, self.s_dim, self.g_repeat_num)
        self.A = Segmentor()

        if torch.cuda.is_available():
            self.G.cuda()
            self.A.cuda()

        self.print_network(self.G, 'G')
        self.print_network(self.A, 'A')

        self.load_test_model()
        self.G.eval()
        self.A.eval()

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def load_test_model(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.test_model))))
        self.A.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_A.pth'.format(self.test_model))))
        print('loaded trained models (step: {})..!'.format(self.test_model))

    def make_celeb_labels_test(self):
        """Generate domain labels for CelebA for debugging/testing.
        """
        fixed_c_list = []
        fixed_c_list.append(self.to_var(torch.FloatTensor([1,0,0,1,1]).unsqueeze(0), volatile=True))
        fixed_c_list.append(self.to_var(torch.FloatTensor([0,1,0,1,1]).unsqueeze(0), volatile=True))
        fixed_c_list.append(self.to_var(torch.FloatTensor([0,0,1,1,1]).unsqueeze(0), volatile=True))
        fixed_c_list.append(self.to_var(torch.FloatTensor([1,0,0,1,0]).unsqueeze(0), volatile=True))
        fixed_c_list.append(self.to_var(torch.FloatTensor([0,1,0,1,0]).unsqueeze(0), volatile=True))
        fixed_c_list.append(self.to_var(torch.FloatTensor([0,0,1,1,0]).unsqueeze(0), volatile=True))
        fixed_c_list.append(self.to_var(torch.FloatTensor([1,0,0,0,1]).unsqueeze(0), volatile=True))
        fixed_c_list.append(self.to_var(torch.FloatTensor([0,1,0,0,1]).unsqueeze(0), volatile=True))
        fixed_c_list.append(self.to_var(torch.FloatTensor([0,0,1,0,1]).unsqueeze(0), volatile=True))
        fixed_c_list.append(self.to_var(torch.FloatTensor([1,0,0,0,0]).unsqueeze(0), volatile=True))
        fixed_c_list.append(self.to_var(torch.FloatTensor([0,1,0,0,0]).unsqueeze(0), volatile=True))
        fixed_c_list.append(self.to_var(torch.FloatTensor([0,0,1,0,0]).unsqueeze(0), volatile=True))

        return fixed_c_list

    def to_var(self, x, volatile=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, volatile=volatile) 

    def test(self):
        fixed_c_list = self.make_celeb_labels_test()
        transform = transforms.Compose([
            transforms.CenterCrop(self.config.celebA_crop_size),
            transforms.Scale(self.config.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_seg1 = transforms.Compose([
            transforms.CenterCrop(self.config.celebA_crop_size),
            transforms.Scale(self.config.image_size)])
        transform_seg2 = transforms.Compose([
            transforms.ToTensor()])

        path = os.path.join(self.config.test_image_path, self.config.test_image_fname)
        image = Image.open(path)
        
        image = transform(image)
        image = image.unsqueeze(0)

        x = self.to_var(image, volatile=True)
        fake_image_mat = []
        for fixed_c in fixed_c_list:
            fake_image_list = [x]

            num_s = 7

            seg = self.A(x)
            seg_numpy = seg.data[0].cpu().float().numpy()
            # seg_numpy = np.transpose(seg_numpy, (1, 2, 0)).astype(np.float)
            seg_max_indices = np.argmax(a, axis=0)

            """ 1-hot encodes a tensor """
            s=np.asarray(seg_max_indices)
            s=np.eye(num_s, dtype='uint8')[s]
            seg_onehot = transform_seg2(seg_onehot)*255.0
            seg_onehot = seg_onehot.unsqueeze(0)    
            s = self.to_var(seg_onehot, volatile=True)

            # import scipy.io as sio
            # sio.savemat('segnumpy.mat',{'seg':seg_numpy})

            fake_x = self.G(x,fixed_c,s)
            fake_image_list.append(fake_x)

            fake_images = torch.cat(fake_image_list, dim=2)
            ake_image_mat.append(fake_images)

        fake_images_save = torch.cat(fake_image_mat, dim=3)
                
        save_path = os.path.join(self.result_path, 'fake_x_sum_{}.png'.format(fname[:-4]))
        print('Translated test images and saved into "{}"..!'.format(save_path))
        save_image(self.denorm(fake_images_save.data), save_path, nrow=1, padding=0)
