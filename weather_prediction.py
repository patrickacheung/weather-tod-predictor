# pylint: disable=missing-docstring, invalid-name, line-too-long
"""
header
"""
import sys
import pandas as pd
from scipy import misc

"""
Returns images as a 1-d array
"""
def get_images(path, filenames):
    # Reading images to df adapted from: https://stackoverflow.com/a/40058708
    imgs = pd.DataFrame(columns=['filename', 'img'])
    for i, fn in enumerate(filenames):
        entry = {'filename': fn, 'img': misc.imread(path+'/'+fn).reshape(-1)}
        entry_df = pd.DataFrame([entry], index=[i])

        imgs = imgs.append(entry_df)

    return imgs


def main():
    in_data_fp = sys.argv[1]
    in_img_fp = sys.argv[2]

    # Only load images that exist from cleaning.
    data = pd.read_csv(in_data_fp)
    imglist = data['filename']
    imgs = get_images(in_img_fp, imglist)

    data_imgs = pd.merge(data, imgs, how='inner', on='filename', sort=False)
    print(data_imgs.head(5))


if __name__ == '__main__':
    main()
