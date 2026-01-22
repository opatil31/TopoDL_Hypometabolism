
addpath('/home/hice1/khom9/scratch/spm')
spm_jobman('initcfg');

matlabbatch{1}.spm.tools.mb.run.mu.create.K = 5;
matlabbatch{1}.spm.tools.mb.run.mu.create.vx = 1;
matlabbatch{1}.spm.tools.mb.run.mu.create.mu_settings = [1e-05 0.5 0];
matlabbatch{1}.spm.tools.mb.run.mu.create.issym = 0;
matlabbatch{1}.spm.tools.mb.run.aff = 'SE(3)';
matlabbatch{1}.spm.tools.mb.run.v_settings = [0.0001 0 0.4 0.1 0.4];
matlabbatch{1}.spm.tools.mb.run.del_settings = Inf;
matlabbatch{1}.spm.tools.mb.run.onam = 'mbb';
matlabbatch{1}.spm.tools.mb.run.odir = {'/tmp/multibrain-mri'};
matlabbatch{1}.spm.tools.mb.run.cat = {{}};
%%
matlabbatch{1}.spm.tools.mb.run.gmm.chan.images = {
                                                   '/home/hice1/khom9/scratch/ad_cohort_mri_flat/ADNI_002_S_0729_MR_MT1__GradWarp__N3m_Br_20120322163605283_S89463_I291876.nii'
                                                   '/home/hice1/khom9/scratch/ad_cohort_mri_flat/ADNI_002_S_0729_MR_MT1__N3m_Br_20120913163818876_S159861_I334105.nii'
                                                   '/home/hice1/khom9/scratch/ad_cohort_mri_flat/ADNI_002_S_1268_MR_MT1__N3m_Br_20120327105530973_S144578_I293691.nii'
                                                   '/home/hice1/khom9/scratch/ad_cohort_mri_flat/ADNI_002_S_4171_MR_MT1__N3m_Br_20121001130644753_S162683_I337465.nii'
                                                   '/home/hice1/khom9/scratch/ad_cohort_mri_flat/ADNI_002_S_4225_MR_MT1__N3m_Br_20140131155655035_S204281_I412309.nii'
                                                   '/home/hice1/khom9/scratch/ad_cohort_mri_flat/ADNI_002_S_5018_MR_MT1__N3m_Br_20121206140426408_S174291_I349888.nii'
                                                   '/home/hice1/khom9/scratch/ad_cohort_mri_flat/ADNI_003_S_1057_MR_MPR__GradWarp__B1_Correction__N3_Br_20090507172646981_S64900_I143414.nii'
                                                   '/home/hice1/khom9/scratch/ad_cohort_mri_flat/ADNI_003_S_1057_MR_MT1__GradWarp__N3m_Br_20120308102117614_S80181_I288889.nii'
                                                   '/home/hice1/khom9/scratch/ad_cohort_mri_flat/ADNI_003_S_1057_MR_MT1__GradWarp__N3m_Br_20120420154238621_S147492_I299336.nii'
                                                   '/home/hice1/khom9/scratch/ad_cohort_mri_flat/ADNI_003_S_1057_MR_MT1__GradWarp__N3m_Br_20160914091135534_S259937_I775696.nii'
                                                   '/home/hice1/khom9/scratch/ad_cohort_mri_flat/ADNI_003_S_1059_MR_MPR__GradWarp__B1_Correction__N3_Br_20070501173419666_S22301_I52811.nii'
                                                   };
%%
matlabbatch{1}.spm.tools.mb.run.gmm.chan.inu.inu_reg = 10000;
matlabbatch{1}.spm.tools.mb.run.gmm.chan.inu.inu_co = 40;
matlabbatch{1}.spm.tools.mb.run.gmm.chan.modality = 1;
matlabbatch{1}.spm.tools.mb.run.gmm.labels.false = [];
matlabbatch{1}.spm.tools.mb.run.gmm.pr.file = {};
matlabbatch{1}.spm.tools.mb.run.gmm.pr.hyperpriors = {
                                                      'b0_priors'
                                                      {
                                                      0.01
                                                      0.01
                                                      }'
                                                      }';
matlabbatch{1}.spm.tools.mb.run.gmm.tol_gmm = 0.0005;
matlabbatch{1}.spm.tools.mb.run.gmm.nit_gmm_miss = 32;
matlabbatch{1}.spm.tools.mb.run.gmm.nit_gmm = 8;
matlabbatch{1}.spm.tools.mb.run.gmm.nit_appear = 8;
matlabbatch{1}.spm.tools.mb.run.accel = 0.8;
matlabbatch{1}.spm.tools.mb.run.min_dim = 8;
matlabbatch{1}.spm.tools.mb.run.tol = 0.001;
matlabbatch{1}.spm.tools.mb.run.sampdens = 2;
matlabbatch{1}.spm.tools.mb.run.save = true;
matlabbatch{1}.spm.tools.mb.run.nworker = 64;

out = spm_jobman('run', matlabbatch);