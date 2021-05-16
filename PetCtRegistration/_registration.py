from MeDas.Alg.common import *
import numpy as np
import os
from scipy.ndimage.interpolation import zoom


class PetCtRegistration(ToolObject):
    """
    @breif PET CT 图像的对齐

    @param ct 是 输入的 CT　图像，输入的影像文件，或者　numpy 矩阵（如果是numpy 矩阵，则需要传递 spacing 和origin 两个参数。
    @param pt 是 输入的 PET 图像，输入的影像文件，或者　numpy 矩阵（如果是numpy 矩阵，则需要传递 spacing 和origin 两个参数。
    @param ct_spacing 手动提供 ct 的 spacing
    @param ct_origin  手动提供 ct 的 origin
    @param pt_spacing 手动提供 pt 的 spacing
    @param pt_origin  手动提供 pt 的 origin
    """

    @staticmethod
    def version():
        return (0,1,0,0,'develop')

    @property
    def inputs(self):
        return [
            ImagePlug('ct', 'ct_header', ct_spacing = "_spacing", ct_origin = "_origin"),
            ImagePlug('pt', 'pt_header', pt_spacing = "_spacing", pt_origin = "_origin")
        ]

    @property
    def output_constructor(self):
        return dict(ct = ImageCons(), pt = ImageCons())


    def kernel(self, ct: np.ndarray, ct_header: FlexHeader, pt: np.ndarray, pt_header: FlexHeader):
        ToolLogger.info('Enter')

        ct_factor = ct_header.spacing
        pt_factor = pt_header.spacing
        ct_local  = ct_header.origin
        pt_local  = pt_header.origin

        scalar = [p / c for (c,p) in zip(ct_factor, pt_factor)]
        scalar.reverse()

        delta  = [c - p for (c,p) in zip(ct_local,  pt_local)]
        offset = (round(delta[0] / ct_factor[0]), round(delta[1] / ct_factor[1]))

        pt_new = zoom(pt, scalar, mode = 'nearest', order = 0)
        pt_new = pt_new[:, offset[1]:offset[1]+ct.shape[1], offset[0]:offset[0]+ct.shape[2]]
        return dict(ct = ct, pt = pt_new)

