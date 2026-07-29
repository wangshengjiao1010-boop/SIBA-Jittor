function evaluate_ssim_variants(evaluation_dir, ir_dir, vi_dir, fused_dir, output_csv)
addpath(evaluation_dir);

ir_files = dir(fullfile(ir_dir, '*'));
ir_files = ir_files(~[ir_files.isdir]);
names = sort({ir_files.name});

image_names = strings(numel(names), 1);
SSIM_uint8 = zeros(numel(names), 1);
SSIM_float255_default = zeros(numel(names), 1);
SSIM_float255_dynamic_range = zeros(numel(names), 1);
MEF_SSIM = zeros(numel(names), 1);

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

    infrared_float = im2double(infrared) * 255.0;
    visible_float = im2double(visible) * 255.0;
    fused_float = im2double(fused) * 255.0;
    image_sequence = zeros(size(infrared, 1), size(infrared, 2), 2);
    image_sequence(:, :, 1) = infrared_float;
    image_sequence(:, :, 2) = visible_float;

    image_names(index) = string(name);
    SSIM_uint8(index) = 0.5 * ssim(fused, infrared) + 0.5 * ssim(fused, visible);
    SSIM_float255_default(index) = 0.5 * ssim(fused_float, infrared_float) + 0.5 * ssim(fused_float, visible_float);
    SSIM_float255_dynamic_range(index) = 0.5 * ssim(fused_float, infrared_float, 'DynamicRange', 255) + 0.5 * ssim(fused_float, visible_float, 'DynamicRange', 255);
    MEF_SSIM(index) = mef_ssim(image_sequence, fused_float);
end

results = table(image_names, SSIM_uint8, SSIM_float255_default, SSIM_float255_dynamic_range, MEF_SSIM);
writetable(results, output_csv);

[output_directory, output_name, ~] = fileparts(output_csv);
summary = table(mean(SSIM_uint8), mean(SSIM_float255_default), mean(SSIM_float255_dynamic_range), mean(MEF_SSIM), ...
    'VariableNames', {'SSIM_uint8', 'SSIM_float255_default', 'SSIM_float255_dynamic_range', 'MEF_SSIM'});
writetable(summary, fullfile(output_directory, strcat(output_name, '_summary.csv')));
end
