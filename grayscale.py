import os
import glob

from skimage import color
from skimage import io
import numpy as np

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


for dset in {"train/","val/"}:
    print(dset)
    for f in {"MK","TH","Nike","Adidas","Puma"}:
        print(f)
        os.chdir(dset+f)
        all_filenames = (i for i in glob.glob('*.{}'.format("jpg")))
        # print("all:", all_filenames)
        for path in all_filenames:
            if not os.path.isdir("/gscale/"):
                os.makedirs("/gscale/")
            
            img = color.rgb2gray(io.imread(path))
            img.save("/gscale/"+path)
            print("/gscale/"+path)
            os.chdir("../..")
            exit()