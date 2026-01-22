import os
import re
import io
import argparse
import pandas as pd
import matlab.engine

from tqdm import tqdm

AD_COHORT_PET_FLAT_DIR = '/home/hice1/khom9/scratch/ad_cohort_pet_coreg'
# AD_COHORT_MRI_DEFORM_DIR = '/home/hice1/khom9/scratch/multibrain-mri-y'
AD_COHORT_MRI_DEFORM_DIR = '/tmp/multibrain-mri-y'



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

    eng = matlab.engine.start_matlab()
    eng.addpath('/home/hice1/khom9/scratch/spm')

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


def apply_all_deformations_pet_parallel(pet_imgs_dir, deformations_dir, out_dir, cohort_info_file, num_engines=4):
    '''
    Parallel version of `apply_all_deformations_pet` that uses multiple concurrent MATLAB engines.
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

    print(f'Starting {num_engines} engines...')
    engines = [matlab.engine.start_matlab() for _ in tqdm(list(range(num_engines)))]
    for eng in engines:
        eng.addpath('/home/hice1/khom9/scratch/spm')

    args = []
    results_async = []
    deform_names = os.listdir(deformations_dir)
    pet_names = os.listdir(pet_imgs_dir)
    output = io.StringIO()

    for i in tqdm(list(range(len(ad_cohort_df)))):
        pet_id, mri_id = pet_ids.iloc[i], mri_ids.iloc[i]
        pet_img_name = [img_name for img_name in pet_names if re.search(rf'^c_({pet_id})', img_name) is not None]
        if len(pet_img_name) != 1:
            continue

        def_name = [img_name for img_name in deform_names if re.search(rf'I({mri_id})', img_name) is not None]
        if len(def_name) != 1:
            # print(mri_id)
            # raise ValueError()
            continue
        pet_img_path = os.path.join(pet_imgs_dir, pet_img_name[0])
        def_path = os.path.join(deformations_dir, def_name[0])

        args.append((engines[i%num_engines], pet_img_path, def_path))
    
    print('Assembled args')
    for eng, pet_img_path, def_path in args:
        out = eng.apply_deformation(pet_img_path, def_path, out_dir, background=True, stdout=output)
        results_async.append(out)

    print('Began all processes')
    results = [x.result() for x in tqdm(results_async, total=len(results_async))]
    for eng in engines:
        eng.quit()        
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parallel', '-p', dest='parallel', action='store_true')
    args = parser.parse_args()

    if args.parallel:
        apply_all_deformations_pet_parallel(
            AD_COHORT_PET_FLAT_DIR,
            AD_COHORT_MRI_DEFORM_DIR,
            '/tmp/ad_cohort_pet_warped',
            '/home/hice1/khom9/CS8903/adni-tables/all_cohorts_pet_mri.csv',
            num_engines=24
        )
    else:
        apply_all_deformations_pet(
            AD_COHORT_PET_FLAT_DIR,
            AD_COHORT_MRI_DEFORM_DIR,
            '/tmp/ad_cohort_pet_warped',
            '/home/hice1/khom9/CS8903/adni-tables/all_cohorts_pet_mri.csv'
        )