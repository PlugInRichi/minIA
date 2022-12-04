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
import argparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from src.minIA.plots import plot_principal_components_2D

parser = argparse.ArgumentParser()
parser.add_argument('descriptors',
                    help='Extractor output file path')
parser.add_argument('cluster',
                    help='Output file path')
parser.add_argument('num_clusters',
                    help='Number of clusters', type=int)
parser.add_argument('-sample_size',
                    help='Sample size for pca', type=int, default=10000)
args = parser.parse_args()
dtype_desc = np.uint8 if ('SIFT' in args.descriptors) else np.float32
path_plot = args.descriptors + '_distribution.jpg'


def main():
    descriptors = np.fromfile(args.descriptors + '.txt', dtype=dtype_desc, count=-1, sep=' ')
    descriptors = descriptors.reshape((int(len(descriptors) / 128), 128))

    print("\nData uploaded successfully, starting clustering...\n")
    kmeans = MiniBatchKMeans(n_clusters=args.num_clusters,
                             init='k-means++',
                             n_init=1,
                             batch_size=4096,
                             verbose=1,
                             max_iter=50,
                             tol=0.000001).fit(descriptors)

    desc_sample = descriptors[np.random.choice(descriptors.shape[0], args.sample_size, replace=False)]
    if 'SIFT' in args.descriptors:
        desc_sample = StandardScaler().fit_transform(desc_sample)
    pca = PCA(n_components=2)
    pc_sample = pca.fit_transform(desc_sample)
    pc_centroids = pca.transform(kmeans.cluster_centers_)
    plot_principal_components_2D(pc_sample, pc_centroids, path_plot)

    del descriptors
    print('Saving data...')
    with open(args.descriptors + '.csv', 'r') as file:
        features_df = pd.read_csv(file)
    features_df['visual_word_id'] = kmeans.labels_.astype(int).tolist()
    features_df.to_csv(args.cluster + '.csv', index=False)
    with open(args.descriptors + '_centroids.txt', 'wb') as file:
        np.savetxt(file, kmeans.cluster_centers_)


if __name__ == '__main__':
    main()
