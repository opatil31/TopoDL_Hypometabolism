function out = dicom_to_nii(dcm_dir, out_dir)
    % out_dir = 'C:\Users\keena\School\Georgia Tech\FA25 Classes\CS 8903\matlab-temp';
    prev_dir = pwd();
    cd(out_dir);
    fnames = spm_select('FPList', dcm_dir, '\.dcm$');
    headers = spm_dicom_headers(fnames, 0);
    out = spm_dicom_convert(headers, 'all', 'patid', 'nii');
    cd(prev_dir);
end

%function nii_names = list_filenames(folder)
 %   
%end


%disp(D.files);
%A = dir(d);
%A = A(~[A.isdir]);

%function fullpath = get_fullpath(listing)
%    fullpath = strcat(listing.folder, "\", listing.name);
%end

%B = arrayfun(@(x) get_fullpath(x), A);

