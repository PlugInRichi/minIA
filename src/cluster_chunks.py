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
parser.add_argument('input_data',
                    help='Extractor output file path')
parser.add_argument('cluster',
                    help='Output file path')
parser.add_argument('num_clusters',
                    help='Number of clusters', type=int)
args = parser.parse_args()


def main():
    kmeans = MiniBatchKMeans(n_clusters=args.num_clusters,
                             init='k-means++',
                             n_init=1,
                             batch_size=4096,
                             verbose=1,
                             max_iter=50,
                             tol=0.000001)

    for i in range(10):
        print("Loading "+str(i)+" chunk for cluster...")
        descriptors = np.fromfile(args.descriptors+'_0'+str(i), dtype=np.float16, count=-1, sep=' ')
        print("Clustering...")
        kmeans.partial_fit(np.reshape(descriptors, (int(len(descriptors)/128), 128)))
        del descriptors
    print('\nCluster done!\nassigning labels for all dataset:\n')
    labels = list()
    for i in range(10):
        print("Loading "+str(i)+" chunk for assign  label...")
        descriptors = np.fromfile(args.descriptors+'_0'+str(i), dtype=np.float16, count=-1, sep=' ')
        labels += kmeans.predict(np.reshape(descriptors, (int(len(descriptors)/128), 128))).astype(int).tolist()
        del descriptors


    print('Saving data...')
    indexes_name = ['index', 'image_name']
    with open(args.input_data + '.csv', 'r') as file:
        features_df = pd.read_csv(file, index_col=indexes_name)
    features_df['descriptor_id'] = labels
    features_df.to_csv(args.cluster + '.csv')


if __name__ == '__main__':
    main()
