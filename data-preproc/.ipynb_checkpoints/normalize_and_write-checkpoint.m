function out = normalize_and_write(img_path, template_path, out_dir)
    prev_dir = pwd();
    cd(out_dir);

    matlabbatch{1}.subj.source = {img_path};
    matlabbatch{1}.subj.wtsrc = '';
    matlabbatch{1}.subj.resample = {img_path};
    matlabbatch{1}.eoptions.template = {template_path};
    matlabbatch{1}.eoptions.weight = '';
    matlabbatch{1}.eoptions.smosrc = 10;
    matlabbatch{1}.eoptions.smoref = 0;
    matlabbatch{1}.eoptions.regtype = 'mni';
    matlabbatch{1}.eoptions.cutoff = 25;
    matlabbatch{1}.eoptions.nits = 16;
    matlabbatch{1}.eoptions.reg = 1;
    matlabbatch{1}.roptions.preserve = 0;
    matlabbatch{1}.roptions.bb = [NaN NaN NaN
                                                                 NaN NaN NaN];
    matlabbatch{1}.roptions.vox = [2 2 2];
    matlabbatch{1}.roptions.interp = 1;
    matlabbatch{1}.roptions.wrap = [0 0 0];
    matlabbatch{1}.roptions.prefix = 'norm_';
    
    out = spm_run_normalise_estwrite(matlabbatch{1});
    cd(prev_dir);
end


