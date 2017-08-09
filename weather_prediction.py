# pylint: disable=missing-docstring, invalid-name, line-too-long
"""
Predicts tomorrow's weather description and temperature based on previous day's
observation data and images.

Outputs analysis graphs.
"""
import sys
import numpy as np
import pandas as pd
from scipy import misc
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

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


"""
Returns array that keeps every 16th pixel.
"""
def reduce_pixels(col):
    return col[1::16]


def main():
    in_data_fp = sys.argv[1]
    in_img_fp = sys.argv[2]

    # Only load images that exist from cleaning.
    data = pd.read_csv(in_data_fp)
    imglist = data['filename']
    imgs = get_images(in_img_fp, imglist)
    data_imgs = pd.merge(data, imgs, how='inner', on='filename', sort=False)

    # Reducing image feature points.
    # Too many features causing out of memory.
    # Adapted from: https://stackoverflow.com/a/17777520
    data_imgs['img'] = data_imgs['img'].apply(reduce_pixels)

    # Exploding flattened 2d image.
    # Adapted from: https://stackoverflow.com/q/32468402
    img_pts = pd.DataFrame(data_imgs['img'].values.tolist())
    data_imgs_pts = data_imgs.join(img_pts)

    # Hard coding 9216 image points from keeping every 16th pixel.
    int_list = list(range(0, 9216))
    #str_list = [str(x) for x in int_list]

    features = [
        'Temp (C)', 'Dew Point Temp (C)', 'Rel Hum (%)', 'Wind Dir (10s deg)', 'Wind Spd (km/h)',
        'Visibility (km)', 'Stn Press (kPa)'#, 'img'
    ] + int_list

    X = data_imgs_pts[features].values
    y = data_imgs_pts['Weather'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = KNeighborsClassifier(n_neighbors=45)
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))
    df = pd.DataFrame({'truth': y_test, 'prediction': model.predict(X_test)})
    print(df[df['truth'] != df['prediction']])


if __name__ == '__main__':
    main()
