from PIL import Image
from os import path
import numpy as np
from glob import glob
import random
import traceback
pwd = path.dirname(__file__)


def disnormalize(arr):
    return (arr*127.5+127.5).astype(np.uint8)


def img2arr(pth, img_siz=128):

    dirpth = path.dirname(pth)
    dirbn = path.basename(dirpth)

    bn = path.basename(pth)
    bn = path.splitext(bn)[0]

    svname = "%s_%s_%dx%d.npy" % (dirbn, bn, img_siz, img_siz)
    svpth = path.join(pwd, 'npy', svname)
    if(path.exists(svpth)):
        return np.load(svpth).astype(np.float32)/127.5-1
    im = Image.open(pth)
    im = im.resize((img_siz, img_siz), Image.LANCZOS).convert("RGB")
    ret = np.asarray(im)

    np.save(svpth, ret)

    return ret.astype(np.float32)/127.5-1

data_pths = None

def get_data_pths():
    global data_pths
    pth = path.join(pwd, 'datas', '*')
    if(data_pths is None):
        data_pths = list(glob(pth))

    return data_pths
def yield_data(n,size=128):
	def sample(ls,n):
		if(len(ls)<n):
			return ls+sample(ls,n-len(ls))
		return random.sample(ls,n)
	pths=get_data_pths()
	pths=sample(pths,n)
	ret=[]
	for pth in pths:
		ret.append(img2arr(pth))
	return ret
def plot_given_img(w, images):
    images = np.array(images)
    images = disnormalize(images)
    img_siz = images.shape[1]
    # print(w,img_siz)
    ret = np.zeros([w*img_siz, w*img_siz, 3], np.uint8)
    for x in range(w):
        for y in range(w):
            idx = x*w+y
            ret[x*img_siz:x*img_siz+img_siz, y *
                img_siz:y*img_siz+img_siz, :] = images[idx]
    return ret

