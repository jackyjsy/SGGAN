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
from model import Discriminator
from model import Segmentor
from PIL import Image
from util.visualizer import Visualizer
import util.util as util
from collections import OrderedDict

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    # print(y)
    # print(y.size())
    y=np.asarray(y)
    # print(type(y))
    y=np.eye(num_classes, dtype='uint8')[y]
    return y

# class CrossEntropyLoss2d(nn.Module):
#     def __init__(self, weight=None, size_average=True, ignore_index=255):
#         super(CrossEntropyLoss2d, self).__init__()
#         self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

#     def forward(self, inputs, targets):
#         ce2d_loss = self.nll_loss(torch.unsqueeze(F.log_softmax(inputs[0]),0), torch.unsqueeze(targets[0],0))
#         for i in range(len(inputs)-1):
#             ce2d_loss = ce2d_loss + self.nll_loss(torch.unsqueeze(F.log_softmax(inputs[i+1]),0),torch.unsqueeze(targets[i+1],0))
#         return ce2d_loss

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        # print(targets.size())
        return self.nll_loss(F.log_softmax(inputs), torch.squeeze(targets))

class Solver(object):

    def __init__(self, celebA_loader, rafd_loader, config):
        # Data loader
        self.celebA_loader = celebA_loader
        self.rafd_loader = rafd_loader
        self.visualizer = Visualizer()
        # Model hyper-parameters
        self.c_dim = config.c_dim
        self.s_dim = config.s_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.d_train_repeat = config.d_train_repeat

        # Hyper-parameteres
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp
        self.lambda_s = config.lambda_s
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.a_lr = config.a_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Criterion
        self.criterion_s = CrossEntropyLoss2d(size_average=True).cuda()

        # Training settings
        self.dataset = config.dataset
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.batch_size = config.batch_size
        self.use_tensorboard = config.use_tensorboard
        self.pretrained_model = config.pretrained_model

        # Test settings
        self.test_model = config.test_model
        self.config = config

        # Path
        self.log_path = config.log_path
        self.sample_path = config.sample_path
        self.model_save_path = config.model_save_path
        self.result_path = config.result_path

        # Step size
        self.log_step = config.log_step
        self.visual_step = self.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step

        # Build tensorboard if use
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()

    def build_model(self):
        # Define a generator and a discriminator
        if self.dataset == 'Both':
            self.G = Generator(self.g_conv_dim, self.c_dim+self.c2_dim+2, self.g_repeat_num)   # 2 for mask vector
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim+self.c2_dim, self.d_repeat_num)
        else:
            self.G = Generator(self.g_conv_dim, self.c_dim, self.s_dim, self.g_repeat_num)
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num) 
            self.A = Segmentor()

        # Optimizers
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.a_optimizer = torch.optim.Adam(self.A.parameters(), self.a_lr, [self.beta1, self.beta2])
        # Print networks
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
        self.print_network(self.A, 'A')

        if torch.cuda.is_available():
            self.G.cuda()
            self.D.cuda()
            self.A.cuda()

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))
        self.D.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_D.pth'.format(self.pretrained_model))))
        self.A.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_A.pth'.format(self.pretrained_model))))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def build_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.log_path)

    def update_lr(self, g_lr, d_lr):
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def to_var(self, x, volatile=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, volatile=volatile)

    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def threshold(self, x):
        x = x.clone()
        x[x >= 0.5] = 1
        x[x < 0.5] = 0
        return x

    def compute_accuracy(self, x, y, dataset):
        if dataset == 'CelebA':
            x = F.sigmoid(x)
            predicted = self.threshold(x)
            correct = (predicted == y).float()
            accuracy = torch.mean(correct, dim=0) * 100.0
        else:
            _, predicted = torch.max(x, dim=1)
            correct = (predicted == y).float()
            accuracy = torch.mean(correct) * 100.0
        return accuracy

    def one_hot(self, labels, dim):
        """Convert label indices to one-hot vector"""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def make_celeb_labels(self, real_c):
        """Generate domain labels for CelebA for debugging/testing.

        if dataset == 'CelebA':
            return single and multiple attribute changes
        elif dataset == 'Both':
            return single attribute changes
        """
        y = [torch.FloatTensor([1, 0, 0]),  # black hair
             torch.FloatTensor([0, 1, 0]),  # blond hair
             torch.FloatTensor([0, 0, 1])]  # brown hair

        fixed_c_list = []

        # single attribute transfer
        for i in range(self.c_dim):
            fixed_c = real_c.clone()
            for c in fixed_c:
                if i < 3:
                    c[:3] = y[i]
                else:
                    c[i] = 0 if c[i] == 1 else 1   # opposite value
            fixed_c_list.append(self.to_var(fixed_c, volatile=True))

        # multi-attribute transfer (H+G, H+A, G+A, H+G+A)
        if self.dataset == 'CelebA':
            for i in range(4):
                fixed_c = real_c.clone()
                for c in fixed_c:
                    if i in [0, 1, 3]:   # Hair color to brown
                        c[:3] = y[2] 
                    if i in [0, 2, 3]:   # Gender
                        c[3] = 0 if c[3] == 1 else 1
                    if i in [1, 2, 3]:   # Aged
                        c[4] = 0 if c[4] == 1 else 1
                fixed_c_list.append(self.to_var(fixed_c, volatile=True))
        return fixed_c_list

    def train(self):
        """Train StarGAN within a single dataset."""

        # Set dataloader
        if self.dataset == 'CelebA':
            self.data_loader = self.celebA_loader
        else:
            self.data_loader = self.rafd_loader

        # The number of iterations per epoch
        iters_per_epoch = len(self.data_loader)

        fixed_x = []
        real_c = []
        fixed_s = []
        for i, (images, seg_i, seg, labels) in enumerate(self.data_loader):
            fixed_x.append(images)
            fixed_s.append(seg)
            real_c.append(labels)
            if i == 3:
                break

        # Fixed inputs and target domain labels for debugging
        fixed_x = torch.cat(fixed_x, dim=0)
        fixed_x = self.to_var(fixed_x, volatile=True)
        real_c = torch.cat(real_c, dim=0)

        
        fixed_s = torch.cat(fixed_s, dim=0)
        fixed_s_list = []
        fixed_s_list.append(self.to_var(fixed_s, volatile=True))
        
        rand_idx = torch.randperm(fixed_s.size(0))
        fixed_s_num = 5
        fixed_s_vec = fixed_s[rand_idx][:fixed_s_num]

        for i in range(fixed_s_num):
            fixed_s_temp = fixed_s_vec[i].unsqueeze(0).repeat(fixed_s.size(0),1,1,1)
            fixed_s_temp = self.to_var(fixed_s_temp)
            fixed_s_list.append(fixed_s_temp)

        # for i in range(4):
        #     rand_idx = torch.randperm(fixed_s.size(0))
        #     fixed_s_temp = self.to_var(fixed_s[rand_idx], volatile=True)
        #     fixed_s_list.append(fixed_s_temp)

        if self.dataset == 'CelebA':
            fixed_c_list = self.make_celeb_labels(real_c)
        elif self.dataset == 'RaFD':
            fixed_c_list = []
            for i in range(self.c_dim):
                fixed_c = self.one_hot(torch.ones(fixed_x.size(0)) * i, self.c_dim)
                fixed_c_list.append(self.to_var(fixed_c, volatile=True))

        # lr cache for decaying
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start with trained model if exists
        if self.pretrained_model:
            start = int(self.pretrained_model.split('_')[0])-1
        else:
            start = 0

        # Start training
        start_time = time.time()
        for e in range(start, self.num_epochs):
            epoch_iter = 0
            for i, (real_x, real_s_i, real_s, real_label) in enumerate(self.data_loader):
                epoch_iter = epoch_iter + 1
                # Generat fake labels randomly (target domain labels)
                rand_idx = torch.randperm(real_label.size(0))
                fake_label = real_label[rand_idx]
                rand_idx = torch.randperm(real_label.size(0))
                fake_s = real_s[rand_idx]
                fake_s_i = real_s_i[rand_idx]
                if self.dataset == 'CelebA':
                    real_c = real_label.clone()
                    fake_c = fake_label.clone()
                else:
                    real_c = self.one_hot(real_label, self.c_dim)
                    fake_c = self.one_hot(fake_label, self.c_dim)

                # Convert tensor to variable
                real_x = self.to_var(real_x)
                real_s = self.to_var(real_s)
                real_s_i = self.to_var(real_s_i)
                fake_s = self.to_var(fake_s)
                fake_s_i = self.to_var(fake_s_i)
                real_c = self.to_var(real_c)           # input for the generator
                fake_c = self.to_var(fake_c)
                real_label = self.to_var(real_label)   # this is same as real_c if dataset == 'CelebA'
                fake_label = self.to_var(fake_label)
                
                # ================== Train D ================== #

                # Compute loss with real images
                out_src, out_cls = self.D(real_x)
                d_loss_real = - torch.mean(out_src)

                if self.dataset == 'CelebA':
                    d_loss_cls = F.binary_cross_entropy_with_logits(
                        out_cls, real_label, size_average=False) / real_x.size(0)
                else:
                    d_loss_cls = F.cross_entropy(out_cls, real_label)

                # Compute classification accuracy of the discriminator
                if (i+1) % self.log_step == 0:
                    accuracies = self.compute_accuracy(out_cls, real_label, self.dataset)
                    log = ["{:.2f}".format(acc) for acc in accuracies.data.cpu().numpy()]
                    if self.dataset == 'CelebA':
                        print('Classification Acc (Black/Blond/Brown/Gender/Aged): ', end='')
                    else:
                        print('Classification Acc (8 emotional expressions): ', end='')
                    print(log)

                # Compute loss with fake images
                fake_x = self.G(real_x, fake_c, fake_s)
                fake_x = Variable(fake_x.data)
                out_src, out_cls = self.D(fake_x)
                d_loss_fake = torch.mean(out_src)

                # Backward + Optimize
                d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Compute gradient penalty
                alpha = torch.rand(real_x.size(0), 1, 1, 1).cuda().expand_as(real_x)
                interpolated = Variable(alpha * real_x.data + (1 - alpha) * fake_x.data, requires_grad=True)
                out, out_cls = self.D(interpolated)

                grad = torch.autograd.grad(outputs=out,
                                           inputs=interpolated,
                                           grad_outputs=torch.ones(out.size()).cuda(),
                                           retain_graph=True,
                                           create_graph=True,
                                           only_inputs=True)[0]

                grad = grad.view(grad.size(0), -1)
                grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                d_loss_gp = torch.mean((grad_l2norm - 1)**2)

                # Backward + Optimize
                d_loss = self.lambda_gp * d_loss_gp
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # ================== Train A ================== #
                self.a_optimizer.zero_grad()
                out_real_s = self.A(real_x)
                # a_loss = self.criterion_s(out_real_s, real_s_i.type(torch.cuda.LongTensor)) * self.lambda_s
                a_loss = self.criterion_s(out_real_s, real_s_i) * self.lambda_s
                # a_loss = torch.mean(torch.abs(real_s - out_real_s))
                a_loss.backward()
                self.a_optimizer.step()

                # Logging
                loss = {}
                loss['D/loss_real'] = d_loss_real.data[0]
                loss['D/loss_fake'] = d_loss_fake.data[0]
                loss['D/loss_cls'] = d_loss_cls.data[0]
                loss['D/loss_gp'] = d_loss_gp.data[0]

                # ================== Train G ================== #
                if (i+1) % self.d_train_repeat == 0:

                    # Original-to-target and target-to-original domain
                    fake_x = self.G(real_x, fake_c, fake_s)

                    rec_x = self.G(fake_x, real_c, real_s)

                    # Compute losses
                    out_src, out_cls = self.D(fake_x)
                    g_loss_fake = - torch.mean(out_src)
                    g_loss_rec = self.lambda_rec * torch.mean(torch.abs(real_x - rec_x))

                    if self.dataset == 'CelebA':
                        g_loss_cls = F.binary_cross_entropy_with_logits(
                            out_cls, fake_label, size_average=False) / fake_x.size(0)
                    else:
                        g_loss_cls = F.cross_entropy(out_cls, fake_label)

                    # segmentation loss
                    out_fake_s = self.A(fake_x)
                    g_loss_s = self.lambda_s * self.criterion_s(out_fake_s, fake_s_i)
                    # Backward + Optimize
                    g_loss = g_loss_fake + g_loss_rec + g_loss_s + self.lambda_cls * g_loss_cls
                    # g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging
                    loss['G/loss_fake'] = g_loss_fake.data[0]
                    loss['G/loss_rec'] = g_loss_rec.data[0]
                    loss['G/loss_cls'] = g_loss_cls.data[0]
                
                if (i+1) % self.visual_step == 0:
                    # save visuals
                    self.real_x = real_x
                    self.fake_x = fake_x
                    self.rec_x = rec_x
                    self.real_s = real_s
                    self.fake_s = fake_s
                    self.out_real_s = out_real_s
                    self.out_fake_s = out_fake_s
                    self.a_loss = a_loss
                    # save losses
                    self.d_real = - d_loss_real
                    self.d_fake = d_loss_fake
                    self.d_loss = d_loss
                    self.g_loss = g_loss
                    self.g_loss_fake = g_loss_fake
                    self.g_loss_rec = g_loss_rec
                    self.g_loss_s = g_loss_s
                    errors_D = self.get_current_errors('D')
                    errors_G = self.get_current_errors('G')
                    self.visualizer.display_current_results(self.get_current_visuals(), e)
                    self.visualizer.plot_current_errors_D(e, float(epoch_iter)/float(iters_per_epoch), errors_D)
                    self.visualizer.plot_current_errors_G(e, float(epoch_iter)/float(iters_per_epoch), errors_G)
                # Print out log info
                if (i+1) % self.log_step == 0:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))

                    log = "Elapsed [{}], Epoch [{}/{}], Iter [{}/{}]".format(
                        elapsed, e+1, self.num_epochs, i+1, iters_per_epoch)

                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

                    if self.use_tensorboard:
                        for tag, value in loss.items():
                            self.logger.scalar_summary(tag, value, e * iters_per_epoch + i + 1)

                # Translate fixed images for debugging
                if (i+1) % self.sample_step == 0:
                    fake_image_list = [fixed_x]
                    fixed_c = fixed_c_list[0]
                    real_seg_list = []
                    for fixed_c in fixed_c_list:
                        for fixed_s in fixed_s_list:
                            fake_image_list.append(self.G(fixed_x, fixed_c, fixed_s))
                            real_seg_list.append(fixed_s)
                    fake_images = torch.cat(fake_image_list, dim=3)
                    real_seg_images = torch.cat(real_seg_list, dim=3)
                    save_image(self.denorm(fake_images.data),
                        os.path.join(self.sample_path, '{}_{}_fake.png'.format(e+1, i+1)),nrow=1, padding=0)
                    save_image(self.cat2class_tensor(real_seg_images.data),
                        os.path.join(self.sample_path, '{}_{}_seg.png'.format(e+1, i+1)),nrow=1, padding=0)
                    print('Translated images and saved into {}..!'.format(self.sample_path))

                # Save model checkpoints
                if (i+1) % self.model_save_step == 0:
                    torch.save(self.G.state_dict(),
                        os.path.join(self.model_save_path, '{}_{}_G.pth'.format(e+1, i+1)))
                    torch.save(self.D.state_dict(),
                        os.path.join(self.model_save_path, '{}_{}_D.pth'.format(e+1, i+1)))
                    torch.save(self.A.state_dict(),
                        os.path.join(self.model_save_path, '{}_{}_A.pth'.format(e+1, i+1)))

            # Decay learning rate
            if (e+1) > (self.num_epochs - self.num_epochs_decay):
                g_lr -= (self.g_lr / float(self.num_epochs_decay))
                d_lr -= (self.d_lr / float(self.num_epochs_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decay learning rate to g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def make_celeb_labels_test(self):
        """Generate domain labels for CelebA for debugging/testing.

        if dataset == 'CelebA':
            return single and multiple attribute changes
        elif dataset == 'Both':
            return single attribute changes
        """
        y = [torch.FloatTensor([1, 0, 0]),  # black hair
             torch.FloatTensor([0, 1, 0]),  # blond hair
             torch.FloatTensor([0, 0, 1])]  # brown hair

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

    def test(self):
        """Facial attribute transfer on CelebA or facial expression synthesis on RaFD."""
        # Load trained parameters
        G_path = os.path.join(self.model_save_path, '{}_G.pth'.format(self.test_model))
        self.G.load_state_dict(torch.load(G_path))
        self.G.eval()
        fixed_c_list = self.make_celeb_labels_test()
        transform = transforms.Compose([
            transforms.CenterCrop(self.config.celebA_crop_size),
            transforms.Scale(self.config.image_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_seg1 = transforms.Compose([
            transforms.CenterCrop(self.config.celebA_crop_size),
            transforms.Scale(self.config.image_size)])
        transform_seg2 = transforms.Compose([
            transforms.ToTensor()])

        for root, _, fnames in sorted(os.walk(self.config.test_image_path)):
            for fname in fnames:
                path = os.path.join(root, fname)
                image = Image.open(path)
        

                image = transform(image)
                image = image.unsqueeze(0)
                x = self.to_var(image, volatile=True)
                fake_image_mat = []
                for fixed_c in fixed_c_list:
                    fake_image_list = [x]
                    for i in range(11):
                        seg = Image.open(os.path.join(self.config.test_seg_path, '{}.png'.format(i+1)))
                        seg = transform_seg1(seg)
                        num_s = 7
                        seg_onehot = to_categorical(seg, num_s)
                        seg_onehot = transform_seg2(seg_onehot)*255.0
                        
                        seg_onehot = seg_onehot.unsqueeze(0)

                        
                        s = self.to_var(seg_onehot, volatile=True)

                        fake_x = self.G(x,fixed_c,s)
                        fake_image_list.append(fake_x)
                        # save_path = os.path.join(self.result_path, 'fake_x_{}.png'.format(i+1))
                        # save_image(self.denorm(fake_x.data), save_path, nrow=1, padding=0)
                    fake_images = torch.cat(fake_image_list, dim=2)
                    fake_image_mat.append(fake_images)

                fake_images_save = torch.cat(fake_image_mat, dim=3)
                
                save_path = os.path.join(self.result_path, 'fake_x_sum_{}.png'.format(fname))
                print('Translated test images and saved into "{}"..!'.format(save_path))
                save_image(self.denorm(fake_images_save.data), save_path, nrow=1, padding=0)
            # # Start translations
            # fake_image_list = [real_x]
            # for target_c in target_c_list:
            #     fake_image_list.append(self.G(real_x, target_c))
            # fake_images = torch.cat(fake_image_list, dim=3)
            # save_path = os.path.join(self.result_path, '{}_fake.png'.format(i+1))
            # save_image(self.denorm(fake_images.data), save_path, nrow=1, padding=0)
            # print('Translated test images and saved into "{}"..!'.format(save_path))
    def test_with_original_seg(self):
        """Facial attribute transfer on CelebA or facial expression synthesis on RaFD."""
        # Load trained parameters
        G_path = os.path.join(self.model_save_path, '{}_G.pth'.format(self.test_model))
        self.G.load_state_dict(torch.load(G_path))
        self.G.eval()
        fixed_c_list = self.make_celeb_labels_test()
        transform = transforms.Compose([
            # transforms.CenterCrop(self.config.celebA_crop_size),
            transforms.Scale(self.config.image_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_seg1 = transforms.Compose([
            transforms.CenterCrop(self.config.celebA_crop_size),
            transforms.Scale(self.config.image_size)])
        transform_seg2 = transforms.Compose([
            transforms.ToTensor()])

        for root, _, fnames in sorted(os.walk(self.config.test_image_path)):
            for fname in fnames:
                path = os.path.join(root, fname)
                image = Image.open(path)
        

                image = transform(image)
                image = image.unsqueeze(0)
                x = self.to_var(image, volatile=True)
                fake_image_mat = []
                for fixed_c in fixed_c_list:
                    fake_image_list = [x]
                    seg = Image.open(os.path.join(self.config.test_seg_path, '{}.png'.format(fname[:-4])))
                    seg = transform_seg1(seg)
                    num_s = 7
                    seg_onehot = to_categorical(seg, num_s)
                    seg_onehot = transform_seg2(seg_onehot)*255.0
                        
                    seg_onehot = seg_onehot.unsqueeze(0)

                    s = self.to_var(seg_onehot, volatile=True)

                    fake_x = self.G(x,fixed_c,s)
                    fake_image_list.append(fake_x)
                    # save_path = os.path.join(self.result_path, 'fake_x_{}.png'.format(i+1))
                    # save_image(self.denorm(fake_x.data), save_path, nrow=1, padding=0)
                    fake_images = torch.cat(fake_image_list, dim=3)
                    fake_image_mat.append(fake_images)

                fake_images_save = torch.cat(fake_image_mat, dim=2)
                
                save_path = os.path.join(self.result_path, 'fake_x_sum_{}.png'.format(fname))
                print('Translated test images and saved into "{}"..!'.format(save_path))
                save_image(self.denorm(fake_images_save.data), save_path, nrow=1, padding=0)
            # # Start translations
            # fake_image_list = [real_x]
            # for target_c in target_c_list:
            #     fake_image_list.append(self.G(real_x, target_c))
            # fake_images = torch.cat(fake_image_list, dim=3)
            # save_path = os.path.join(self.result_path, '{}_fake.png'.format(i+1))
            # save_image(self.denorm(fake_images.data), save_path, nrow=1, padding=0)
            # print('Translated test images and saved into "{}"..!'.format(save_path))
    def test_seg(self):
        """Facial attribute transfer on CelebA or facial expression synthesis on RaFD."""
        # Load trained parameters
        A_path = os.path.join(self.model_save_path, '{}_A.pth'.format(self.test_model))
        self.A.load_state_dict(torch.load(A_path))
        self.A.eval()

        transform = transforms.Compose([
            # transforms.CenterCrop(self.config.celebA_crop_size),
            transforms.Scale(self.config.image_size),
            # transforms.Scale(178),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        for root, _, fnames in sorted(os.walk(self.config.test_image_path)):
            for fname in fnames:
                path = os.path.join(root, fname)
                image = Image.open(path)
                print('Read image "{}"..!'.format(fname))

                image = transform(image)
                image = image.unsqueeze(0)
                x = self.to_var(image, volatile=True)
                
                seg = self.A(x)
                seg_numpy = seg.data[0].cpu().float().numpy()
                seg_numpy = np.transpose(seg_numpy, (1, 2, 0)).astype(np.float)
                import scipy.io as sio
                sio.savemat('segnumpy.mat',{'seg':seg_numpy})
                
                print('Translated seg images and saved into "{}"..!'.format('segnumpy.mat'))
                
    def get_current_errors(self, label='all'):
        D_fake = self.d_fake.data[0]
        D_real = self.d_real.data[0]
        # D_fake = self.D_fake.data[0]
        # D_real = self.D_real.data[0]
        A_loss = self.a_loss.data[0]
        D_loss = self.d_loss.data[0]
        G_loss = self.g_loss.data[0]
        G_loss_fake = self.g_loss_fake.data[0]
        G_loss_s = self.g_loss_s.data[0]
        G_loss_rec = self.g_loss_rec.data[0]
        if label == 'all':
            return OrderedDict([('D_fake', D_fake), 
                                ('D_real', D_real), 
                                ('D', D_loss),
                                ('A_loss', A_loss),
                                ('G', G_loss), 
                                ('G_loss_fake', G_loss_fake), 
                                ('G_loss_s', G_loss_s),
                                ('G_loss_rec', G_loss_rec)])
        if label == 'D':
            return OrderedDict([('D_fake', D_fake), 
                                ('D_real', D_real), 
                                ('D', D_loss),
                                ('A_loss', A_loss)])
        if label == 'G':
            return OrderedDict([('A_loss', A_loss),
                                ('G', G_loss), 
                                ('G_loss_fake', G_loss_fake), 
                                ('G_loss_s', G_loss_s),
                                ('G_loss_rec', G_loss_rec)])

    def get_current_visuals(self):
        real_x = util.tensor2im(self.real_x.data)
        fake_x = util.tensor2im(self.fake_x.data)
        rec_x = util.tensor2im(self.rec_x.data)
        real_s = util.tensor2im_seg(self.real_s.data)
        fake_s = util.tensor2im_seg(self.fake_s.data)
        out_real_s = util.tensor2im_seg(self.out_real_s.data)
        out_fake_s = util.tensor2im_seg(self.out_fake_s.data)
        return OrderedDict([('real_x', real_x), 
                            ('fake_x', fake_x), 
                            ('rec_x', rec_x), 
                            ('real_s', self.cat2class(real_s)),
                            ('fake_s', self.cat2class(fake_s)),
                            ('out_real_s', self.cat2class(out_real_s)), 
                            ('out_fake_s', self.cat2class(out_fake_s))
                            ])
    def cat2class(self, m):
        y = np.zeros((np.size(m,0),np.size(m,1)),dtype='float64')
        for i in range(np.size(m,2)):
            y = y + m[:,:,i]*i
        y = y / float(np.max(y)) * 255.0 
        y = y.astype(np.uint8)
        y = np.reshape(y,(np.size(m,0),np.size(m,1),1))
        # print(np.shape(y))
        return np.repeat(y, 3, 2)

    def cat2class_tensor(self, m):
        y = []
        for i in range(m.size(0)):
            x = torch.cuda.FloatTensor(m.size(2),m.size(3)).zero_()
            for j in range(m.size(1)):
                x = x + m[i,j,:,:]*j
            x = x.unsqueeze(0).unsqueeze(1).expand(1,3,m.size(2),m.size(3))
            y.append(x)
        y = torch.cat(y, dim=0)
        y = y / float(torch.max(y))
        return y
            

