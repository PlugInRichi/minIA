import numpy as np
import cv2 as cv
from custom_delf.delf_extractor import ExtractorDELF


class Extractor(object):
    def __init__(self):
        self.model = None
        self.filter = False
        self.feature_dim = 0
    def read_path(self, image_path):
        if self.filter:
            img = cv.imread(image_path, cv.COLOR_HSV2BGR)
            image_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img = cv.medianBlur(image_gray, 9)
        else:
            img = cv.imread(image_path, cv.COLOR_BGR2GRAY)
        return img

    def get_features(self, image_path):
        imagen = self.read_path(image_path)
        kps, descriptors = self.model.detectAndCompute(imagen, None)
        if descriptors is not None:
            kp_location = [np.rint((kp.pt[0], kp.pt[1])) for kp in kps]
            kp_size = [kp.size for kp in kps]
            return {'location': kp_location, 'size': kp_size}, descriptors
        else:
            return None, None


class Sift(Extractor):
    def __init__(self, auto, nfeatures, nOctaveLayers, contrastThreshold,
                 edgeThreshold, sigma):
        super().__init__()
        self.feature_dim = 128
        if auto:
            self.model = cv.xfeatures2d_SIFT.create()
        else:
            self.model = cv.xfeatures2d_SIFT.create(nfeatures, nOctaveLayers,
            contrastThreshold, edgeThreshold, sigma)


class Surf(Extractor):
    def __init__(self, auto, threshold, nOctaves, nOctaveLayers, extended,
                upright):
        super().__init__()
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
        descriptors = extracted_features['descriptors']
        if len(descriptors) > 0:
            locations = extracted_features['locations']
            feature_scales = extracted_features['scales']
            score = extracted_features['attention']
            return {'location': locations, 'size': feature_scales, 'score': score}, descriptors
        else:
            return None

