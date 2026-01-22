function out = apply_deformation(img_path, deformation_path, out_dir)
    spm_jobman('initcfg');

    matlabbatch{1}.spm.util.defs.comp{1}.def = {deformation_path};
    matlabbatch{1}.spm.util.defs.out{1}.push.fnames = {img_path};
    matlabbatch{1}.spm.util.defs.out{1}.push.weight = {''};
    matlabbatch{1}.spm.util.defs.out{1}.push.savedir.saveusr = {out_dir};
    matlabbatch{1}.spm.util.defs.out{1}.push.fov.bbvox.bb = [NaN NaN NaN
                                                             NaN NaN NaN];
    matlabbatch{1}.spm.util.defs.out{1}.push.fov.bbvox.vox = [NaN NaN NaN];
    matlabbatch{1}.spm.util.defs.out{1}.push.preserve = 0;
    matlabbatch{1}.spm.util.defs.out{1}.push.fwhm = [0 0 0];
    matlabbatch{1}.spm.util.defs.out{1}.push.prefix = 'ww';

    out = spm_jobman('run', matlabbatch);
end
