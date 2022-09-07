import numpy as np
import cv2 as cv
from src.delf.delf_extractor import ExtractorDELF


class Extractor(object):
    def __int__(self):
        self.model = None
        self.filter = False
    def read_path(self, image_path):
        if self.filter:
            img = cv.imread(image_path, cv.COLOR_HSV2BGR)
            image_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img = cv.medianBlur(image_gray, args.median_value)
        else:
            img = cv.imread(image_path, cv.COLOR_BGR2GRAY)
        return img

    def get_features(self, image_path):
        imagen = self.read_path(image_path)
        kps, descs = self.model.detectAndCompute(imagen,None)
        keypoints = list()
        for kp in kps:
            keypoints.append(np.array([kp.pt[0], kp.pt[1], kp.size]))
        return  {'keypoints': keypoints, 'descriptors':descs}


class Sift(Extractor):
    def __init__(self, auto, nfeatures, nOctaveLayers, contrastThreshold,
                 edgeThreshold, sigma):
        if auto:
            self.model = cv.xfeatures2d_SIFT.create()
        else:
            self.model = cv.xfeatures2d_SIFT.create(nfeatures, nOctaveLayers,
            contrastThreshold, edgeThreshold, sigma)


class Surf(Extractor):
    def __init__(self, auto, threshold, nOctaves, nOctaveLayers, extended,
                upright):
        if auto:
            self.model = cv.xfeatures2d.SURF_create()
        else:
            self.model = cv.xfeatures2d.SURF_create(threshold, nOctaves,
            nOctaveLayers, extended, upright)


class Delf(Extractor):
    def __init__(self, delf_configuration):
        self.model = ExtractorDELF(delf_configuration)

    def get_features(self, image_path):
        extracted_features = self.model.extract_features(image_path)
        locations = extracted_features['locations']
        descriptors = extracted_features['descriptors']
        feature_scales = extracted_features['scales']
        attention = extracted_features['attention']
        key_points = np.concatenate((locations, feature_scales.reshape(-1, 1)), axis=1)
        return {'keypoints': key_points,
                'descriptors':descriptors,
                'attention':attention}
