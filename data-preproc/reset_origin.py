# import pydicom as dcm
import nibabel as nib
# import pandas as pd
import numpy as np
import os
import matlab.engine

from tqdm import tqdm

from adni_util import get_fullscan


OFFSET = np.array([118.5, 120, -72])


def get_nii_affine(nii_path):
    '''
    Given the path of the NIFTI file, return only the affine matrix
    '''
    img = nib.load(nii_path)
    return img.affine


def get_dcm_paths(cohort_dir):  
    '''Generate the paths to the DICOM files in the ADNI data directory'''  
    for patient_id in os.listdir(cohort_dir):
        patient_id_path = os.path.join(cohort_dir,patient_id)
        for img_preproc_type in os.listdir(patient_id_path):
            img_preproc_path = os.path.join(patient_id_path, img_preproc_type)
            for visit_date in os.listdir(img_preproc_path):
                visit_date_path = os.path.join(img_preproc_path, visit_date)
                for img_id in os.listdir(visit_date_path):
                    dcm_path = os.path.join(visit_date_path, img_id)
                    yield dcm_path


def convert_all_dicom(cohort_dir, nii_dir):
    '''
    Convert all the images in a cohort from individual DICOM slices into NIFTI files (.nii).
    Images are stored in nested folders in the format {patient_id}/{preprocess_type}/{img_id}.nii
    '''
    eng = matlab.engine.start_matlab()
    eng.cd(os.getcwd())
    for path in tqdm(list(get_dcm_paths(cohort_dir))):
        out = eng.dicom_to_nii(path, nii_dir, nargout=1)
    eng.quit()


def reset_all_origins(cohort_nii_dir, out_dir):
    '''Reset the origins for all images to the center (near ACM). Images are not nested in subfolders'''

    for patient_id in tqdm(os.listdir(cohort_nii_dir)):
        patient_id_path = os.path.join(cohort_nii_dir, patient_id)
        for preproc in os.listdir(patient_id_path):
            preproc_path = os.path.join(patient_id_path, preproc)
            for nii_name in os.listdir(preproc_path):
                nii_path = os.path.join(preproc_path, nii_name)
                img = nib.load(nii_path)
                img.affine[:3, 3] += OFFSET
                
                img.header.set_sform(img.affine)
                # img.header.set_qform(img.affine)
                nib.save(img, os.path.join(out_dir, nii_name))


def normalize_to_template(nii_dir, template_path, out_dir):
    '''
    Warp all images in the directory to the specified template. The images should be in NIFTI1 format, and 
    their origins should overlap with the template within +/- 4cm and +/- 15 degrees
    '''
    eng = matlab.engine.start_matlab()
    eng.cd(os.getcwd())
    for img_name in tqdm(os.listdir(nii_dir)):
        out = eng.normalize_and_write(os.path.join(img_name, nii_dir), template_path, out_dir)
    eng.quit()





def check_same_affine(cohort_nii_dir):
    affines = []
    for nii_name in os.listdir(cohort_nii_dir):
        affine = get_nii_affine(os.path.join(cohort_nii_dir, nii_name))
        affines.append(affine)

    # True means all affine matrices are equal
    return np.all([np.allclose(affines[0], affines[i]) for i in range(len(affines))])



if __name__ == '__main__':
    ad_cohort_dir = r'C:\Users\keena\School\Georgia Tech\FA25 Classes\CS 8903\Hierarchical Clustering\AD_Cohort'
    ad_cohort_nii_dir = r'C:\Users\keena\School\Georgia Tech\FA25 Classes\CS 8903\data-preproc\ad_cohort_nii'
    ad_cohort_nii_reset_dir = r'C:\Users\keena\School\Georgia Tech\FA25 Classes\CS 8903\data-preproc\ad_cohort_nii_reset'
    ad_cohort_warp_dir = r'C:\Users\keena\School\Georgia Tech\FA25 Classes\CS 8903\data-preproc\ad_cohort_warp'

    # Convert ADNI DICOM data (list of .dcm) into NIFTI files (.nii)
    convert_all_dicom(ad_cohort_dir, ad_cohort_nii_dir)

    # Check that all the images have the same affine matrix (i.e., same origin and voxel size)
    check_same_affine(ad_cohort_nii_dir)

    # Reset the origins of all images to the center of the image (80, 80, 48)
    reset_all_origins(ad_cohort_nii_dir, ad_cohort_nii_reset_dir)

    # Warp all images in the cohort to the template and save
    normalize_to_template(ad_cohort_nii_reset_dir, ad_cohort_warp_dir)