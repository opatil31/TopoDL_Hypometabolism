import os
import re
import pandas as pd
import matlab.engine

from tqdm import tqdm

def coreg_all_images(pet_imgs_dir, mri_imgs_dir, out_dir, cohort_info_file):
    '''
    Co-register the PET images to corresponding MRI images. Each PET image should be matched with
    an MRI image to register it to. This is given by a DataFrame stored in `cohort_info_file`.

    Each image directory should be `flat`.
    '''
    if not os.path.exists(pet_imgs_dir):
        raise ValueError(f'PET image folder does not exist: {pet_imgs_dir}')
    if not os.path.exists(mri_imgs_dir):
        raise ValueError(f'MRI image folder does not exist: {mri_imgs_dir}')
    if not os.path.isfile(cohort_info_file):
        raise ValueError(f'Cohort info file does not exist: {cohort_info_file}')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        print(f'Created new folder "{out_dir}"')

    cohort_df = pd.read_csv(cohort_info_file)
    ad_cohort_df = cohort_df[cohort_df['DIAGNOSIS'] == 3.0]
    pet_ids, mri_ids = ad_cohort_df['image_id_pet'], ad_cohort_df['image_id_mri']

    eng = matlab.engine.start_matlab()
    eng.addpath('/home/hice1/khom9/scratch/spm')
    eng.cd(out_dir)

    mri_names = os.listdir(mri_imgs_dir)
    pet_names = os.listdir(pet_imgs_dir)

    for i in tqdm(list(range(len(ad_cohort_df)))):
        pet_id, mri_id = pet_ids.iloc[i], mri_ids.iloc[i]
        pet_img_name = [img_name for img_name in pet_names if re.search(rf'^({pet_id})', img_name) is not None]
        if len(pet_img_name) != 1:
            continue

        mri_img_name = [img_name for img_name in mri_names if re.search(rf'I({mri_id})', img_name) is not None]
        if len(mri_img_name) != 1:
            # print(mri_id)
            # raise ValueError()
            continue

        pet_img_path = os.path.join(pet_imgs_dir, pet_img_name[0])
        pet_img_path = os.path.join(mri_imgs_dir, mri_img_name[0])

        out = eng.apply_deformation(pet_img_path, pet_img_path, out_dir)