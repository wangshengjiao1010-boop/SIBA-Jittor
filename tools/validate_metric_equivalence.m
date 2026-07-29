function validate_metric_equivalence(evaluation_dir, infrared_path, visible_path, fused_path, output_csv)
addpath(evaluation_dir);
infrared = imread(infrared_path);
visible = imread(visible_path);
fused = imread(fused_path);
if size(infrared, 3) > 2
    infrared = rgb2gray(infrared);
end
if size(visible, 3) > 2
    visible = rgb2gray(visible);
end
if size(fused, 3) > 2
    fused = rgb2gray(fused);
end

infrared = im2double(infrared) * 255;
visible = im2double(visible) * 255;
fused = im2double(fused) * 255;
sequence = zeros(size(infrared, 1), size(infrared, 2), 2);
sequence(:, :, 1) = infrared;
sequence(:, :, 2) = visible;

tic;
mef_official = mef_ssim(sequence, fused);
mef_official_seconds = toc;
tic;
mef_fast = mef_ssim_fast(sequence, fused);
mef_fast_seconds = toc;
tic;
ms_official = analysis_ms_ssim(sequence, fused);
ms_official_seconds = toc;
tic;
ms_fast = analysis_ms_ssim_fast(sequence, fused);
ms_fast_seconds = toc;
tic;
fmi_official = analysis_fmi(infrared, visible, fused);
fmi_official_seconds = toc;
tic;
fmi_fast = analysis_fmi_fast(infrared, visible, fused);
fmi_fast_seconds = toc;

results = table( ...
    mef_official, mef_fast, abs(mef_official - mef_fast), mef_official_seconds, mef_fast_seconds, ...
    ms_official, ms_fast, abs(ms_official - ms_fast), ms_official_seconds, ms_fast_seconds, ...
    fmi_official, fmi_fast, abs(fmi_official - fmi_fast), fmi_official_seconds, fmi_fast_seconds, ...
    'VariableNames', { ...
        'MEF_SSIM_official', 'MEF_SSIM_fast', 'MEF_SSIM_abs_error', 'MEF_SSIM_official_seconds', 'MEF_SSIM_fast_seconds', ...
        'MS_SSIM_official', 'MS_SSIM_fast', 'MS_SSIM_abs_error', 'MS_SSIM_official_seconds', 'MS_SSIM_fast_seconds', ...
        'FMI_official', 'FMI_fast', 'FMI_abs_error', 'FMI_official_seconds', 'FMI_fast_seconds'});
writetable(results, output_csv);
end
