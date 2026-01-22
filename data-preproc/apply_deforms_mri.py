import os
import io
import matlab.engine
from pathlib import Path
from tqdm import tqdm

# AD_COHORT_DIR = r'C:\Users\keena\School\Georgia Tech\FA25 Classes\CS 8903\cVAE\AD_Cohort_MRI'

AD_COHORT_FLAT_DIR = '/home/hice1/khom9/scratch/ad_cohort_mri_flat'
# AD_COHORT_FLAT_DIR = '/tmp/ADNI_Data/cVAE/ad_cohort_mri_flat'
# AD_COHORT_DEFORM_DIR = '/home/hice1/khom9/scratch/multibrain-mri-y-small'
AD_COHORT_DEFORM_DIR = '/tmp/multibrain-mri-y'

def apply_all_deformations_mri(imgs_dir, deformations_dir, out_dir):
    '''
    Apply deformations to the given sMRI images. Writes the result to `out_dir`.
    `imgs_dir` should be "flat", that is, all images stored in one folder with no subfolders.  
    '''
    if not os.path.exists(imgs_dir):
        raise ValueError(f'Image folder does not exist: {imgs_dir}')
    if not os.path.exists(deformations_dir):
        raise ValueError(f'Deformation folder does not exist: {deformations_dir}')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        print(f'Created new folder "{out_dir}"')
    
    eng = matlab.engine.start_matlab()
    eng.cd(os.getcwd())

    deformations_names = os.listdir(deformations_dir)

    for img_name in tqdm(os.listdir(imgs_dir)):
        def_fields_names = [x for x in deformations_names if Path(img_name).stem in x and x.startswith('y_')]
        if len(def_fields_names) != 1:
            continue
        #     raise ValueError(f'Should only find one deformation field for "{img_name}. Found {len(def_fields)}."')
        img_path = os.path.join(imgs_dir, img_name)
        def_path = os.path.join(deformations_dir, def_fields_names[0])

        out = eng.apply_deformation(img_path, def_path, out_dir)


def apply_all_deformations_mri_parallel(imgs_dir, deformations_dir, out_dir, num_engines=4):
    '''
    Parallel version of `apply_all_deformations_mri` that uses multiple concurrent MATLAB engines.
    '''
    if not os.path.exists(imgs_dir):
        raise ValueError(f'Image folder does not exist: {imgs_dir}')
    if not os.path.exists(deformations_dir):
        raise ValueError(f'Deformation folder does not exist: {deformations_dir}')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        print(f'Created new folder "{out_dir}"')
    
    print(f'Starting {num_engines} engines...')
    engines = [matlab.engine.start_matlab() for _ in tqdm(list(range(num_engines)))]
    for eng in engines:
        eng.addpath('/home/hice1/khom9/scratch/spm')

    args = []
    results_async = []
    output = io.StringIO()
    
    deformations_names = os.listdir(deformations_dir)
    
    for i,img_name in enumerate(os.listdir(imgs_dir)):
        def_fields_names = [x for x in deformations_names if Path(img_name).stem in x and x.startswith('y_')]
        if len(def_fields_names) != 1:
            continue
        #     raise ValueError(f'Should only find one deformation field for "{img_name}. Found {len(def_fields)}."')
        img_path = os.path.join(imgs_dir, img_name)
        def_path = os.path.join(deformations_dir, def_fields_names[0])

        args.append((engines[i%num_engines], img_path, def_path))

    print('Assembled args')
    for eng, img_path, def_path in args:
        out = eng.apply_deformation(img_path, def_path, out_dir, background=True, stdout=output)
        results_async.append(out)
        
    print('Began all processes')
    results = [x.result() for x in tqdm(results_async, total=len(results_async))]
    for eng in engines:
        eng.quit()

    



if __name__ == '__main__':
    deform_out_dir = '/tmp/ad_cohort_mri_warped'
    apply_all_deformations_mri_parallel(
        AD_COHORT_FLAT_DIR, 
        AD_COHORT_DEFORM_DIR, 
        deform_out_dir,
        num_engines=24
        )