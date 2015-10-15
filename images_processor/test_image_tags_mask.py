# coding: utf-8
import h5py
import numpy as np
from images_processor import ImagesProcessor

fname = "/home/sala/Work/Data/SACLA/259408_roi.h5"
f = h5py.File(fname)

tags = f["run_259408/event_info/tag_number_list"][:]
tags_mask = np.zeros(tags.shape, dtype=bool)
tags_mask[0::2] = True
dset = f["run_259408/detector_2d_1"]
spectra_ref = [dset["tag_%s/detector_data" % str(tag)][:].sum(axis=1) for tag in tags[tags_mask]]

ip = ImagesProcessor("SACLA")
ip.add_analysis("get_projection", args={'axis': 1, })
ip.set_dataset("/run_259408/detector_2d_1")
results = ip.analyze_images(fname, n=-1, tags=tags[tags_mask])
spectra = results['get_projection']['spectra']

print (spectra - spectra_ref == 0).all()
