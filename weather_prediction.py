# pylint: disable=missing-docstring, invalid-name, line-too-long
"""
header
"""
import sys
import pandas as pd
from scipy import misc

def main():
    in_data_fp = sys.argv[1]
    in_img_fp = sys.argv[2]

    data = pd.read_csv(in_data_fp)
    imglist = data['filename']
    print(imglist)


if __name__ == '__main__':
    main()
