# pylint: disable=missing-docstring, invalid-name, line-too-long
"""
Predicts the image's current weather category.
Categories are Clear, Cloudy, Fog, Rain, Snow, Thunderstorm.
Generates graphs.
"""
import sys
import seaborn
import pandas as pd
import matplotlib.pyplot as plt
from scipy import misc
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
Returns array that keeps every 8th pixel.
"""
def reduce_pixels(col):
    return col[1::8]


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

    # Hard coding 18432 image points from keeping every 8th pixel.
    int_list = list(range(0, 18432))

    features = [
        'Temp (C)', 'Dew Point Temp (C)', 'Rel Hum (%)', 'Wind Dir (10s deg)', 'Wind Spd (km/h)',
        'Visibility (km)', 'Stn Press (kPa)'#, 'img'
    ] + int_list

    # Weather category prediction.
    X = data_imgs_pts[features].values
    y = data_imgs_pts['Weather'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model_category = make_pipeline(
        StandardScaler(),
        PCA(500),
        SVC(kernel='linear', C=5)
    )
    model_category.fit(X_train, y_train)

    df = pd.DataFrame({'truth': y_test, 'prediction': model_category.predict(X_test)})
    df = df[df['truth'] != df['prediction']]

    ctgy_counts = df.groupby('prediction').count().reset_index()
    xticks_labels = ctgy_counts['prediction'].values

    plt.figure(figsize=(10, 7))
    plt.suptitle('Weather Description Incorrect Counts')
    plt.ylabel('Count')
    plt.xlabel('Category')
    plt.xticks(ctgy_counts.index.values, xticks_labels)
    plt.bar(ctgy_counts.index.values, ctgy_counts['truth'], align='center', alpha=0.5)
    seaborn.set()
    plt.savefig('ctgy_count.png')

    print('category score: ', model_category.score(X_test, y_test))

    # ToD prediction.
    X_tod = X
    y_tod = data_imgs_pts['Time'].values

    X_tod_train, X_tod_test, y_tod_train, y_tod_test = train_test_split(X_tod, y_tod)

    model_tod = make_pipeline(
        StandardScaler(),
        PCA(500),
        SVC(kernel='linear', C=1000)
    )
    model_tod.fit(X_tod_train, y_tod_train)

    df_tod = pd.DataFrame({'truth': y_tod_test, 'prediction': model_tod.predict(X_tod_test)})
    df_tod = df_tod[df_tod['truth'] != df_tod['prediction']]

    df_tod['prediction'] = pd.to_datetime(df_tod['prediction'])
    df_tod['truth'] = pd.to_datetime(df_tod['truth'])
    df_tod['diff (hrs)'] = df_tod['prediction'].dt.hour - df_tod['truth'].dt.hour

    tod_counts = df_tod.groupby('diff (hrs)').count().reset_index()
    plt.figure(figsize=(10, 7))
    plt.suptitle('Difference Between Predicted Time and Actual Time')
    plt.ylabel('Count')
    plt.xlabel('Difference (hours)')
    plt.bar(tod_counts['diff (hrs)'], tod_counts['truth'], align='center', alpha=0.5)
    seaborn.set()
    plt.savefig('tod_count.png')

    print('tod score: ', model_tod.score(X_tod_test, y_tod_test))


if __name__ == '__main__':
    main()
