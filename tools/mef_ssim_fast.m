function [Q, qMap] = mef_ssim_fast(imgSeq, fI, K, window)
if nargin < 2 || nargin > 4
    Q = -Inf;
    qMap = Inf;
    return;
end
if ~exist('K', 'var')
    K = 0.03;
end
if ~exist('window', 'var')
    window = fspecial('gaussian', 11, 1.5);
end

imgSeq = double(imgSeq);
fI = double(fI);
[~, ~, source_count] = size(imgSeq);
window_size = size(window, 1);
sample_count = window_size^2;
square_window = ones(window_size) / sample_count;

mu = zeros(size(imgSeq, 1) - window_size + 1, size(imgSeq, 2) - window_size + 1, source_count);
ed = zeros(size(mu));
uniform_covariance = cell(source_count, source_count);
gaussian_mu = zeros(size(mu));
gaussian_covariance = cell(source_count, source_count);

for source_index = 1:source_count
    source = imgSeq(:, :, source_index);
    mu(:, :, source_index) = conv2(source, square_window, 'valid');
    variance = conv2(source .* source, square_window, 'valid') - mu(:, :, source_index).^2;
    ed(:, :, source_index) = sqrt(max(sample_count * variance, 0)) + 0.001;
    gaussian_mu(:, :, source_index) = conv2(source, window, 'valid');
end

for first_index = 1:source_count
    first_source = imgSeq(:, :, first_index);
    for second_index = 1:source_count
        second_source = imgSeq(:, :, second_index);
        uniform_covariance{first_index, second_index} = ...
            conv2(first_source .* second_source, square_window, 'valid') - ...
            mu(:, :, first_index) .* mu(:, :, second_index);
        gaussian_covariance{first_index, second_index} = ...
            conv2(first_source .* second_source, window, 'valid') - ...
            gaussian_mu(:, :, first_index) .* gaussian_mu(:, :, second_index);
    end
end

source_sum = sum(imgSeq, 3);
source_sum_mu = conv2(source_sum, square_window, 'valid');
source_sum_variance = conv2(source_sum .* source_sum, square_window, 'valid') - source_sum_mu.^2;
numerator = sqrt(max(sample_count * source_sum_variance, 0));
denominator = sum(ed - 0.001, 3);
consistency = (numerator + eps) ./ (denominator + eps);
consistency(consistency > 1) = 1 - eps;
consistency(consistency < 0) = eps;

power_map = tan(pi / 2 * consistency);
power_map(power_map > 10) = 10;
weight_map = (ed / window_size) .^ repmat(power_map, [1, 1, source_count]) + eps;
weight_map = weight_map ./ repmat(sum(weight_map, 3), [1, 1, source_count]);
alpha = weight_map ./ ed;

raw_norm_squared = zeros(size(consistency));
raw_variance = zeros(size(consistency));
for first_index = 1:source_count
    for second_index = 1:source_count
        coefficient = alpha(:, :, first_index) .* alpha(:, :, second_index);
        raw_norm_squared = raw_norm_squared + sample_count * coefficient .* uniform_covariance{first_index, second_index};
        raw_variance = raw_variance + coefficient .* gaussian_covariance{first_index, second_index};
    end
end
raw_norm = sqrt(max(raw_norm_squared, 0));
scale = zeros(size(raw_norm));
nonzero = raw_norm > 0;
max_ed = max(ed, [], 3);
scale(nonzero) = max_ed(nonzero) ./ raw_norm(nonzero);

fused_mu = conv2(fI, window, 'valid');
fused_variance = conv2(fI .* fI, window, 'valid') - fused_mu.^2;
raw_fused_covariance = zeros(size(consistency));
for source_index = 1:source_count
    source = imgSeq(:, :, source_index);
    covariance = conv2(source .* fI, window, 'valid') - gaussian_mu(:, :, source_index) .* fused_mu;
    raw_fused_covariance = raw_fused_covariance + alpha(:, :, source_index) .* covariance;
end

sigma1_squared = scale.^2 .* max(raw_variance, 0);
sigma2_squared = max(fused_variance, 0);
sigma12 = scale .* raw_fused_covariance;
constant = (K * 255)^2;
qMap = (2 * sigma12 + constant) ./ (sigma1_squared + sigma2_squared + constant);
Q = mean(qMap, 'all');
end
