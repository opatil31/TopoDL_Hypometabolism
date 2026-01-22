import pydicom as pdc
import os
import numpy as np
import nibabel as nib

from multiprocessing import Pool
from tqdm import tqdm


def get_fullscan(dcm_dir, dtype=None):
    '''
    Given the directory storing all DCM files, get the 3D tensor containing all of its data with shape 
    (num_slices, rows, cols)
    '''
    fullscan = np.empty(1)
    patient_id = None
    acq_date = None
    
    for i,fname in enumerate(os.listdir(dcm_dir)):
        ds = pdc.dcmread(os.path.join(dcm_dir, fname))

        if i == 0:
            patient_id = ds.PatientID
            acq_date = ds.AcquisitionDate
            fullscan = np.empty((ds.NumberOfSlices, ds.Rows, ds.Columns), dtype=np.float32 if dtype is None else dtype)
        try:
            fullscan[ds.ImageIndex-1] = ds.pixel_array
        except:
            fullscan[ds.InstanceNumber-1] = ds.pixel_array
    return fullscan, patient_id, acq_date


def get_patient_data_parallel(args):
#     patient_id_path = os.path.join(cohort_dir, patient_id)
    imgs, img_ids, patient_info = [], [], []
    
    patient_id_path, normalize = args['patient_id_path'], args['normalize']
    for img_preproc_type in os.listdir(patient_id_path):
        img_preproc_path = os.path.join(patient_id_path, img_preproc_type)
        for visit_date in os.listdir(img_preproc_path):
            visit_date_path = os.path.join(img_preproc_path, visit_date)
            for img_id in os.listdir(visit_date_path):
#                     img_ids.append(img_id[1:])
                dcm_path = os.path.join(visit_date_path, img_id)
                img, patient_id, acq_date = mean_normalize(get_fullscan(dcm_path)) \
                                            if normalize \
                                            else get_fullscan(dcm_path)
#                     mci_imgs.append(img)
#                     patient_info.append((patient_id, acq_date))
                imgs.append(img) 
                img_ids.append(img_id[1:])
                patient_info.append((patient_id, acq_date))
    return np.stack(imgs), img_ids, patient_info


def get_cohort_data(cohort_dir, normalize=False):
    '''
    Get image data for a given cohort. ADNI's download format is expected
    '''
                        
    with Pool() as pool:
        args = [{
            'patient_id_path': os.path.join(cohort_dir, patient_id),
            'normalize': normalize
        } for patient_id in os.listdir(cohort_dir)]
        res = list(tqdm(pool.imap(get_patient_data_parallel, args, chunksize=5), total=len(args)))
        
        imgs, img_ids, patient_info = list(zip(*res))
#     for patient_id in tqdm(os.listdir(cohort_dir)):
    
    return np.concatenate(imgs, axis=0), np.concatenate(img_ids), np.concatenate(patient_info)


def get_normalized_data(norm_nii_dir, normalize=False):
    '''
    Get PET scan data that has been normalized to a common space.
    '''
    imgs = []
    img_ids = []
    patient_info = []
    for img_name in os.listdir(norm_nii_dir):
        img = nib.load(os.path.join(norm_nii_dir, img_name))
        img_name_split = img_name.lstrip('norm_').split('-')
        img_id, patient_id = img_name_split[:2]
        
        imgs.append(mean_normalize(img.get_fdata()) if normalize else img.get_fdata())
        img_ids.append(img_id)
        patient_info.append(patient_id)
        
    return np.array(imgs), np.array(img_ids), np.array(patient_info)


def mean_normalize(img):
    '''Globally scale an entire scan by dividing it by its mean'''
    return img / img.mean()

