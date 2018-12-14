#encoding=utf-8

import clr
import numpy as np
import PIL.Image as Image
import os

clr.FindAssembly('./PupReader.dll')
clr.FindAssembly('./NetTopologySuite.dll')
from PupReader import Read19Product
from PupReader.Product import BaseReflectivity0

def height(x,y,array):
    return array[x,y]

def Make36AnglePNG(rdf_19):
    print(rdf_19)
    reader19 = Read19Product(rdf_19)
    array_allangle = np.array(list(reader19.Product.AllAngleDcrData), dtype=np.uint8)
    array_3d = array_allangle.reshape((36, 460, 460))
    for i in range(36):
        img_a = Image.fromarray(array_3d[i, :, :], 'L')
        imgfile_a = os.path.join(r'radartemp', r'temp.%d.png' % (i))
        img_a.save(imgfile_a)
    return reader19

def Make36AngleImgArray(rdf_19):
    print(rdf_19)
    reader19 = Read19Product(rdf_19)
    array_allangle = np.array(list(reader19.Product.AllAngleDcrData), dtype=np.uint8)
    array_3d = array_allangle.reshape((36, 460, 460))
    imgs = []
    angles = []
    for i in range(36):
        img_L= Image.fromarray(array_3d[i, :, :], 'L')
        imgs.append(img_L.convert('RGB'))
        angles.append(i*10)
    return reader19, imgs, angles



