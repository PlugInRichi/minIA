#!/usr/bin/env python
# coding: utf-8

"""
Create a visual vocabulary

Input:
    List of descriptors created by extractor
    Number of clusters
Output:
    Visual of visual word identifiers associated with the descriptors
"""

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('descriptors',
                    help='Extractor output file path')
parser.add_argument('cluster',
                    help='Output file path')
parser.add_argument('num_clusters',
                    help='Number of clusters', type=int)
args = parser.parse_args()
dtype_desc = np.uint8 if ('SIFT' in args.descriptors) else np.float32


def main():

    descriptors = np.loadtxt(args.descriptors + '.txt', dtype=dtype_desc)

    print("\nData uploaded successfully, starting clustering...\n")
    kmeans = MiniBatchKMeans(n_clusters=args.num_clusters,
                             init='k-means++',
                             n_init=1,
                             batch_size=4096,
                             verbose=1,
                             max_iter=50,
                             tol=0.000001).fit(descriptors)
    del descriptors
    print('Saving data...')
    with open(args.descriptors + '.csv', 'r') as file:
        features_df = pd.read_csv(file)
    features_df['descriptor_id'] = kmeans.labels_.astype(int).tolist()
    features_df.to_csv(args.cluster + '.csv')


if __name__ == '__main__':
    main()
