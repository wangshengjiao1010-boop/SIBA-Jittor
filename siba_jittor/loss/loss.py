import numpy as np
import jittor as jt
from jittor import nn


def laplacian(input, kernel_size, border_type='reflect', normalized=True):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    kernel_height, kernel_width = kernel_size
    kernel = np.ones((kernel_height, kernel_width), dtype=np.float32)
    kernel[kernel_height // 2, kernel_width // 2] = 1 - kernel.sum()
    if normalized:
        kernel = kernel / np.abs(kernel).sum()
    kernel = jt.array(kernel).cast(input.dtype).reshape((1, 1, kernel_height, kernel_width))
    kernel = kernel.repeat(input.shape[1], 1, 1, 1)
    padding = (kernel_width // 2, kernel_width // 2, kernel_height // 2, kernel_height // 2)
    input = nn.pad(input, padding, mode=border_type)
    return nn.conv2d(input, kernel, groups=input.shape[1])

'''
From CHITNet: A Complementary to Harmonious Information Transfer Network for Infrared and Visible Image Fusion
'''
class JointGrad(nn.Module):
    def __init__(self):
        super(JointGrad, self).__init__()
        self.laplacian = laplacian
        self.l1_loss = nn.L1Loss()

    def execute(self, im_fus, im_ir, im_vi):
        ir_grad = jt.abs(self.laplacian(im_ir, 3))
        vi_grad = jt.abs(self.laplacian(im_vi, 3))
        fus_grad = self.laplacian(im_fus, 3)
        JGrad = jt.where(ir_grad-vi_grad >= 0, self.laplacian(im_ir, 3), self.laplacian(im_vi, 3))
        loss_JGrad = self.l1_loss(JGrad, fus_grad)
        return loss_JGrad


'''
From CDDFuse: Correlation-Driven Dual-Branch Feature Decomposition for Multi-Modality Image Fusion
'''
class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy()
        
    def execute(self,generate_img,image_ir,image_vis):
        
        # int calculation
        image_y=image_vis[:,:1,:,:]
        x_in_max=jt.maximum(image_y,image_ir)
        loss_in=nn.l1_loss(x_in_max,generate_img)

        # grad calculation
        y_grad=self.sobelconv(image_y)
        ir_grad=self.sobelconv(image_ir)
        generate_img_grad=self.sobelconv(generate_img)
        x_grad_joint=jt.maximum(y_grad,ir_grad)
        loss_grad=nn.l1_loss(x_grad_joint,generate_img_grad)
        
        return loss_in, loss_grad


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = jt.array(kernelx).float32().unsqueeze(0).unsqueeze(0)
        kernely = jt.array(kernely).float32().unsqueeze(0).unsqueeze(0)
        self.weightx = kernelx.stop_grad()
        self.weighty = kernely.stop_grad()

    def execute(self,x):

        sobelx=nn.conv2d(x, self.weightx, padding=1)
        sobely=nn.conv2d(x, self.weighty, padding=1)
        return jt.abs(sobelx)+jt.abs(sobely)
