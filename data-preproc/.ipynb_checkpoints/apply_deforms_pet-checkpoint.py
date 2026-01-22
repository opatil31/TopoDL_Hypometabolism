import os
import re
import pandas as pd
import matlab.engine

from tqdm import tqdm

AD_COHORT_PET_FLAT_DIR = '/home/hice1/khom9/scratch/ad_cohort_nii_reset_small'
AD_COHORT_MRI_DEFORM_DIR = '/home/hice1/khom9/scratch/multibrain-mri-y'


def apply_all_deformations_pet(pet_imgs_dir, deformations_dir, out_dir, cohort_info_file):
    '''
    Apply deformations obtained from multibrain fit to PET images. Writes the results to `out_dir`.
    `pet_imgs_dir` should be "flat", that is, all images stored in one folder with no subfolders.  

    `cohort_info_file` is a CSV that should match each PET image with its closest MRI image in time for each 
    given patient. There should be an `image_id_pet` column and `image_id_mri` column.
    '''
    if not os.path.exists(pet_imgs_dir):
        raise ValueError(f'Image folder does not exist: {pet_imgs_dir}')
    if not os.path.exists(deformations_dir):
        raise ValueError(f'Deformation folder does not exist: {deformations_dir}')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        print(f'Created new folder "{out_dir}"')

    cohort_df = pd.read_csv(cohort_info_file)
    ad_cohort_df = cohort_df[cohort_df['DIAGNOSIS'] == 3.0]
    pet_ids, mri_ids = ad_cohort_df['image_id_pet'], ad_cohort_df['image_id_mri']

    display(pet_ids)
    display(mri_ids)

    eng = matlab.engine.start_matlab()
    eng.cd(os.getcwd())

    deform_names = os.listdir(deformations_dir)
    pet_names = os.listdir(pet_imgs_dir)

    for i in tqdm(list(range(len(ad_cohort_df)))):
        pet_id, mri_id = pet_ids.iloc[i], mri_ids.iloc[i]
        pet_img_name = [img_name for img_name in pet_names if re.search(rf'^({pet_id})', img_name) is not None]
        if len(pet_img_name) != 1:
            continue

        def_name = [img_name for img_name in deform_names if re.search(rf'I({mri_id})', img_name) is not None]
        if len(def_name) != 1:
            # print(mri_id)
            # raise ValueError()
            continue

        pet_img_path = os.path.join(pet_imgs_dir, pet_img_name[0])
        def_path = os.path.join(deformations_dir, def_name[0])

        out = eng.apply_deformation(pet_img_path, def_path, out_dir)
        
        
        
if __name__ == '__main__':
    apply_all_deformations_pet(
        AD_COHORT_PET_FLAT_DIR,
        AD_COHORT_MRI_DEFORM_DIR,
        '/home/hice1/khom9/scratch/ad_cohort_pet_warped',
        '/home/hice1/khom9/CS8903/adni-tables/all_cohorts_pet_mri.csv'
    )