import jittor as jt
import numpy as np

# from DePF: A Novel Fusion Approach based on Decomposition Pooling for Infrared and Visible Images
def RGB2YCrCb(rgb_image):
    """
    RGB to YCrCb
    """
    R = rgb_image[0:1]
    G = rgb_image[1:2]
    B = rgb_image[2:3]

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = clamp(Y)
    Cr = clamp(Cr)
    Cb = clamp(Cb)
    return Y, Cb, Cr


def YCrCb2RGB(Y, Cb, Cr):
    """
    YcrCb to RGB
    """
    ycrcb = jt.concat([Y, Cr, Cb], dim=0)
    C, W, H = ycrcb.shape
    im_flat = ycrcb.reshape(3, -1).transpose(0, 1)
    mat = jt.array(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cast(Y.dtype)
    bias = jt.array([0.0 / 255, -0.5, -0.5]).cast(Y.dtype)
    temp = jt.matmul(im_flat + bias, mat)
    out = temp.transpose(0, 1).reshape(C, W, H)
    out = clamp(out)
    return out


def clamp(value, min=0., max=1.0):
    """
    force the pixel values to be constrained within [0,1] to avoid abnormal spots
    """
    if isinstance(value, np.ndarray):
        return np.clip(value, min, max)
    return jt.clamp(value, min, max)
