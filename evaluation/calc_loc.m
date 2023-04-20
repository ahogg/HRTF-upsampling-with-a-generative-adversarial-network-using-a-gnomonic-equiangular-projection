function [pol_acc1, pol_rms1, querr1] = calc_loc(fullSofa1FileName, fullSofa2FileName)

    curdir = cd; amt_start(); cd(curdir); % start AMT
    pol_acc1_avg = 0.0;
    pol_rms1_avg = 0.0;
    querr1_avg = 0.0;
    Sofa1 = SOFAload(fullSofa1FileName);
    Sofa2 = SOFAload(fullSofa2FileName);
    [h1,fs,az,el] = sofa2hrtf(Sofa1);
    [h2,fs2,az2,el2] = sofa2hrtf(Sofa2);
    fs = 48000;
    fs2 = 48000;
    num_exp = 1;

    % Run barumerli2021 for h1
    disp('Running barumerli2021 for first HRTF...'), tic

    dtf = getDTF(h1,fs);
    SOFA_obj1 = hrtf2sofa(dtf,fs,az,el);
    % Preprocessing source information
    [~, target1] = barumerli2021_featureextraction(SOFA_obj1, 'dtf', 'targ_az', SOFA_obj1.SourcePosition(:, 1), 'targ_el', SOFA_obj1.SourcePosition(:, 2)); 


    dtf = getDTF(h2,fs2);
    SOFA_obj2 = hrtf2sofa(dtf,fs2,az2,el2);
    % Preprocessing source information
    [template2, target2] = barumerli2021_featureextraction(SOFA_obj2, 'dtf', 'targ_az', SOFA_obj2.SourcePosition(:, 1), 'targ_el', SOFA_obj2.SourcePosition(:, 2));

    % Run virtual experiments
    [m1,doa1] = barumerli2021('template',template2,'target',target1,'num_exp',num_exp,'sigma_spectral', 4, 'MAP');
    % Calculate performance measures
    sim1 = barumerli2021_metrics(m1, 'middle_metrics');
    lat_acc1 = sim1.accL; % mean lateral error
    lat_rms1 = sim1.rmsL; % lateral RMS error
    pol_acc1 = sim1.accP; % mean polar error
    pol_rms1 = sim1.rmsP; % polar RMS error
    querr1 = sim1.querr; % quadrant error percentage 
    pol_acc1_avg = pol_acc1_avg + pol_acc1;
    pol_rms1_avg = pol_rms1_avg + pol_rms1;
    querr1_avg = querr1_avg + querr1;
    fprintf('Finished running barumerli2021 fo HRTF.')

end