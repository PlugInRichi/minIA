import numpy as np
import tensorflow as tf

from google.protobuf import text_format
from delf import delf_config_pb2
from delf import utils
from custom_delf.utils import custom_extractor as extractor

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

class ExtractorDELF:

    def __init__(self, config_path):
        config = delf_config_pb2.DelfConfig()
        with tf.io.gfile.GFile(config_path, 'r') as f:
            text_format.Merge(f.read(), config)
        self.model = extractor.MakeExtractor(config)

    def extract_features(self, image_path):
        # Extract features.
        im = np.array(utils.RgbLoader(image_path))
        extracted_features = self.model(im)
        return  extracted_features['local_features']
