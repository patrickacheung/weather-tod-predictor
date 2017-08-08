# pylint: disable=missing-docstring, invalid-name, line-too-long
"""
Cleans data to be more usable for prediction/learning.
Outputs csv file with matching observations and image filename.
"""
import glob
import re
import sys
import pandas as pd
from scipy import misc

# Date/Time regex: yyyy-mm-dd hh:mm
filename_re = re.compile(r'(\d{4})-(\d{2})-(\d{02}) (\d{02}):(\d{02})')

# Img filename regex: katkam-yyyymmddhhmmss.jpg
img_fn_re = re.compile(r'[a-z]{6}-\d{14}.jpg')

# Weather description regex.
# Adapted regex from: https://stackoverflow.com/a/3040797
w_rgx = ['Clear', 'Cloudy', 'Rain', 'Fog', 'Snow', 'Thunderstorm', 'Drizzle']
combined = "(" + ")|(".join(w_rgx) + ")" # Middle repeats.
ctgy_re = re.compile(combined)

"""
Joins multiple csv files into a single dataframe.
"""
def get_joined_data(in_dir):
    # Read multiple .csv files.
    # Adapted from: # Adapted from: https://stackoverflow.com/a/21232849
    all_files = glob.glob(in_dir + '/*.csv')
    observations = pd.DataFrame()
    for _, _file in enumerate(all_files):
        data = pd.read_csv(_file, skiprows=16) # Important info starts at line 17.
        observations = observations.append(data)

    return observations


"""
Cleans and adds columns to make data useable.
"""
def pre_process(obs):
    subset = [
        'Date/Time', 'Temp (°C)', 'Dew Point Temp (°C)', 'Rel Hum (%)',
        'Wind Dir (10s deg)', 'Wind Spd (km/h)', 'Visibility (km)',
        'Stn Press (kPa)', 'Weather'
    ]
    obs.dropna(how='any', subset=subset, inplace=True)

    # Unused properties.
    drop_list = [
        'Data Quality',
        'Temp Flag',
        'Dew Point Temp Flag',
        'Rel Hum Flag',
        'Wind Dir Flag',
        'Wind Spd Flag',
        'Visibility Flag',
        'Stn Press Flag',
        'Hmdx Flag',
        'Wind Chill Flag'
    ]
    obs.drop(drop_list, axis=1, inplace=True)

    # Add filename to df based on Date/Time.
    obs['filename'] = obs['Date/Time'].apply(add_filename)

    # Sort by Date/Time, ascending.
    obs.sort_values(by='Date/Time', inplace=True)

    return obs


"""
Adds filename to dataframe based on Date/Time.
Filename is katkam-yyyymmddhhmmss.jpg
"""
def add_filename(col):
    match = filename_re.search(col)

    # Brackets for indentation.
    fn = ('katkam-' + match.group(1) + match.group(2) + match.group(3) +
          match.group(4) + match.group(5) + '00.jpg')

    return fn


"""
Returns filename and images.
"""
def get_imgs(in_dir):
    # Reading images to df adapted from: https://stackoverflow.com/a/40058708
    all_imgs = glob.glob(in_dir + '/*.jpg')
    imgs = pd.DataFrame(columns=['filename', 'img'])
    for i, _img in enumerate(all_imgs):
        match = img_fn_re.search(_img)

        entry = {'filename': match.group(0), 'img': misc.imread(_img).reshape(-1)}
        entry_df = pd.DataFrame([entry], index=[i])

        imgs = imgs.append(entry_df)

    return imgs


"""
Returns filename of images.
"""
def get_imgs_fns(in_dir):
    # Reading images to df adapted from: https://stackoverflow.com/a/40058708
    all_imgs = glob.glob(in_dir + '/*.jpg')
    imgs = pd.DataFrame(columns=['filename'])
    for i, _img in enumerate(all_imgs):
        match = img_fn_re.search(_img)

        df = pd.DataFrame({'filename': match.group(0)}, index=[i])
        imgs = imgs.append(df)

    return imgs


"""
Cleans up description so they are more concrete
"""
def clean_descrip(col):
    matches = ctgy_re.findall(col)

    # Adapted list comprehension from: https://stackoverflow.com/a/40598337
    # and https://stackoverflow.com/a/10941335
    if matches:
        # a = [(t), (t, t, t), (t)] where list is a and tuple is t
        # x is iter
        deslist = [x for t in matches for x in t if x]

        # Replace 'Drizzle' with 'Rain' and remove dups
        deslist[:] = [w.replace('Drizzle', 'Rain') for w in deslist]
        # Dict comprehension: https://stackoverflow.com/a/1747827
        no_dups_deslist = {k:0 for k in deslist}.keys()
        return ','.join(no_dups_deslist)

    return col


def main():
    in_dir_obs = sys.argv[1]
    in_dir_img = sys.argv[2]

    obs = get_joined_data(in_dir_obs)
    obs_clean = pre_process(obs)
    #imgs = get_imgs(in_dir_img)
    imgs = get_imgs_fns(in_dir_img)

    obs_imgs = pd.merge(obs_clean, imgs, how='inner', on='filename', sort=False)
    # Clean up some headers.
    obs_imgs = obs_imgs.rename(columns={'Temp (°C)': 'Temp (C)', 'Dew Point Temp (°C)': 'Dew Point Temp (C)'})
    # Clean up 'Weather' column so descriptions are more concrete.
    obs_imgs['Weather'] = obs_imgs['Weather'].apply(clean_descrip)

    #print(obs_imgs['Weather'].unique())
    obs_imgs.to_csv('weather_data.csv', index=False)


if __name__ == '__main__':
    main()
