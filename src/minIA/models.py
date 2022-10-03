import numpy as np
import cv2 as cv
from custom_delf.delf_extractor import ExtractorDELF


class Extractor(object):
    def __init__(self):
        self.model = None
        self.filter = False
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
            descriptors = descriptors.astype('int32')
            kp_location = [np.array((kp.pt[0], kp.pt[1])).round() for kp in kps]
            kp_size = [kp.size for kp in kps]
            return {'location': kp_location, 'size': kp_size, 'descriptor': descriptors.tolist()}
        else:
            return None


class Sift(Extractor):
    def __init__(self, auto, nfeatures, nOctaveLayers, contrastThreshold,
                 edgeThreshold, sigma):
        super().__init__()
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
        locations = extracted_features['locations']
        descriptors = extracted_features['descriptors']
        descriptors = descriptors if len(descriptors) > 0 else None
        feature_scales = extracted_features['scales']
        score = extracted_features['attention']
        return {'location': locations, 'size': feature_scales, 'descriptor': descriptors, 'score': score}
