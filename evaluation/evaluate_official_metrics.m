function evaluate_official_metrics(evaluation_dir, ir_dir, vi_dir, fused_dir, output_csv)
addpath(evaluation_dir);

ir_files = dir(fullfile(ir_dir, '*'));
ir_files = ir_files(~[ir_files.isdir]);
names = {ir_files.name};
names = sort(names);

image_names = strings(numel(names), 1);
VIF = zeros(numel(names), 1);
SCD = zeros(numel(names), 1);
MI = zeros(numel(names), 1);
Qabf = zeros(numel(names), 1);
SSIM = zeros(numel(names), 1);
MS_SSIM = zeros(numel(names), 1);
FMI = zeros(numel(names), 1);

for index = 1:numel(names)
    name = names{index};
    infrared = imread(fullfile(ir_dir, name));
    visible = imread(fullfile(vi_dir, name));
    fused = imread(fullfile(fused_dir, name));
    if size(infrared, 3) > 2
        infrared = rgb2gray(infrared);
    end
    if size(visible, 3) > 2
        visible = rgb2gray(visible);
    end
    if size(fused, 3) > 2
        fused = rgb2gray(fused);
    end
    if ~isequal(size(infrared), size(visible), size(fused))
        error('Image size mismatch: %s', name);
    end

    infrared_float = im2double(infrared) * 255.0;
    visible_float = im2double(visible) * 255.0;
    fused_float = im2double(fused) * 255.0;
    image_sequence = zeros(size(infrared, 1), size(infrared, 2), 2);
    image_sequence(:, :, 1) = infrared_float;
    image_sequence(:, :, 2) = visible_float;

    image_names(index) = string(name);
    VIF(index) = vifp_mscale(infrared_float, fused_float) + vifp_mscale(visible_float, fused_float);
    SCD(index) = analysis_SCD(infrared_float, visible_float, fused_float);
    MI(index) = MI_evaluation(infrared, visible, fused, 256);
    Qabf(index) = analysis_Qabf(infrared_float, visible_float, fused_float);
    SSIM(index) = mef_ssim_fast(image_sequence, fused_float);
    MS_SSIM(index) = analysis_ms_ssim_fast(image_sequence, fused_float);
    FMI(index) = analysis_fmi_fast(infrared_float, visible_float, fused_float);
end

results = table(image_names, VIF, SCD, MI, Qabf, SSIM, MS_SSIM, FMI);
writetable(results, output_csv);

[output_directory, output_name, ~] = fileparts(output_csv);
summary = table(mean(VIF), mean(SCD), mean(MI), mean(Qabf), mean(SSIM), mean(MS_SSIM), mean(FMI), ...
    'VariableNames', {'VIF', 'SCD', 'MI', 'Qabf', 'SSIM', 'MS_SSIM', 'FMI'});
writetable(summary, fullfile(output_directory, strcat(output_name, '_summary.csv')));
end
