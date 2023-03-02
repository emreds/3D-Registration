from collections import defaultdict


class DatasetConfig:
    model_folder = 'models'
    train_folder = 'train'
    test_folder = 'test'
    img_folder = 'rgb'
    depth_folder = 'depth'
    img_ext = 'png'
    depth_ext = 'png'


config = defaultdict(lambda *_: DatasetConfig())

config['tless'] = tless = DatasetConfig()
tless.model_folder = 'models_cad'
tless.test_folder = 'test_primesense'
tless.train_folder = 'train_primesense'

config['tless3'] = tless3 = DatasetConfig()
tless3.model_folder = 'models_cad'
tless3.test_folder = 'test_primesense'
tless3.train_folder = 'train_pbr'
tless3.img_ext = 'jpg'

config['tless3_non_textured'] = tless3_non_textured = DatasetConfig()
tless3_non_textured.model_folder = 'models_cad'
tless3_non_textured.test_folder = 'test_primesense'
tless3_non_textured.train_folder = 'train_pbr'
tless3_non_textured.img_ext = 'jpg'

config['tlesstextured01'] = tlesstextured01 = DatasetConfig()
tlesstextured01.model_folder = 'models_cad'
tlesstextured01.test_folder = 'test_primesense'
tlesstextured01.train_folder = 'train_pbr'
tlesstextured01.img_ext = 'jpg'

config['tlesstextured02'] = tlesstextured02 = DatasetConfig()
tlesstextured02.model_folder = 'models_cad'
tlesstextured02.test_folder = 'test_primesense'
tlesstextured02.train_folder = 'train_pbr'
tlesstextured02.img_ext = 'jpg'

config['tlesstextured03'] = tlesstextured03 = DatasetConfig()
tlesstextured03.model_folder = 'models_cad'
tlesstextured03.test_folder = 'test_primesense'
tlesstextured03.train_folder = 'train_pbr'
tlesstextured03.img_ext = 'jpg'

config['tlesstextured04'] = tlesstextured04 = DatasetConfig()
tlesstextured04.model_folder = 'models_cad'
tlesstextured04.test_folder = 'test_primesense'
tlesstextured04.train_folder = 'train_pbr'
tlesstextured04.img_ext = 'jpg'

config['tlesstextured05'] = tlesstextured05 = DatasetConfig()
tlesstextured05.model_folder = 'models_cad'
tlesstextured05.test_folder = 'test_primesense'
tlesstextured05.train_folder = 'train_pbr'
tlesstextured05.img_ext = 'jpg'

config['hb'] = hb = DatasetConfig()
hb.test_folder = 'test_primesense'

config['itodd'] = itodd = DatasetConfig()
itodd.depth_ext = 'tif'
itodd.img_folder = 'gray'
itodd.img_ext = 'tif'
