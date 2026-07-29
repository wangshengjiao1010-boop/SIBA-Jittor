function nfmi = analysis_fmi_fast(ima, imb, imf, feature, w)
if nargin < 3
    error('There should be three input images (2 source images and 1 fused image)!');
end
if ~isequal(size(ima), size(imb))
    error('Size of the source images must be the same!');
end
if ~isequal(size(ima), size(imf))
    error('Size of the source and fused images must be the same!');
end
if ~exist('feature', 'var')
    feature = 'edge';
end
if ~exist('w', 'var')
    w = 3;
end

ima = double(ima);
imb = double(imb);
imf = double(imf);
switch feature
    case 'none'
        a_feature = ima;
        b_feature = imb;
        f_feature = imf;
    case 'gradient'
        a_feature = gradient(ima);
        b_feature = gradient(imb);
        f_feature = gradient(imf);
    case 'edge'
        a_feature = edge(ima);
        b_feature = edge(imb);
        f_feature = edge(imf);
    case 'dct'
        a_feature = dct2(ima);
        b_feature = dct2(imb);
        f_feature = dct2(imf);
    case 'wavelet'
        [cA, cH, cV, cD] = dwt2(ima, 'dmey');
        a_feature = rerange_fast([cA, cH; cV, cD]);
        [cA, cH, cV, cD] = dwt2(imb, 'dmey');
        b_feature = rerange_fast([cA, cH; cV, cD]);
        [cA, cH, cV, cD] = dwt2(imf, 'dmey');
        f_feature = rerange_fast([cA, cH; cV, cD]);
    otherwise
        error('Please specify a supported feature extraction method.');
end

window_size = 2 * floor(w / 2) + 1;
fused_patches = im2col(f_feature, [window_size, window_size], 'sliding');
a_patches = im2col(a_feature, [window_size, window_size], 'sliding');
b_patches = im2col(b_feature, [window_size, window_size], 'sliding');
fmi_a = pair_fmi(a_patches, fused_patches);
fmi_b = pair_fmi(b_patches, fused_patches);
nfmi = mean((fmi_a + fmi_b) / 2);
end

function values = pair_fmi(source_patches, fused_patches)
patch_count = size(source_patches, 2);
values = zeros(1, patch_count);
chunk_size = 20000;
for chunk_start = 1:chunk_size:patch_count
    chunk_end = min(chunk_start + chunk_size - 1, patch_count);
    source = double(source_patches(:, chunk_start:chunk_end));
    fused = double(fused_patches(:, chunk_start:chunk_end));
    equal = all(source == fused, 1);
    chunk_values = ones(1, size(source, 2));
    active = ~equal;
    if any(active)
        chunk_values(active) = pair_fmi_active(source(:, active), fused(:, active));
    end
    values(chunk_start:chunk_end) = chunk_values;
end
end

function values = pair_fmi_active(source, fused)
source = normalize_patches(source);
fused = normalize_patches(fused);
source_pdf = source ./ sum(source, 1);
fused_pdf = fused ./ sum(fused, 1);
source_cdf = cumsum(source_pdf, 1);
fused_cdf = cumsum(fused_pdf, 1);

source_centered = source_pdf - mean(source_pdf, 1);
fused_centered = fused_pdf - mean(fused_pdf, 1);
dot_product = sum(source_centered .* fused_centered, 1);
correlation = zeros(size(dot_product));
nonzero_dot = dot_product ~= 0;
correlation(nonzero_dot) = dot_product(nonzero_dot) ./ sqrt( ...
    sum(source_centered(:, nonzero_dot).^2, 1) .* sum(fused_centered(:, nonzero_dot).^2, 1));

positions = (1:size(source_pdf, 1))';
source_mean = sum(positions .* source_pdf, 1);
fused_mean = sum(positions .* fused_pdf, 1);
source_sd = sqrt(sum((positions.^2) .* source_pdf, 1) - source_mean.^2);
fused_sd = sqrt(sum((positions.^2) .* fused_pdf, 1) - fused_mean.^2);

positive = correlation >= 0;
negative = ~positive;
upper_covariance = zeros(size(correlation));
lower_covariance = zeros(size(correlation));
for row_index = 1:size(source_pdf, 1)
    for column_index = 1:size(source_pdf, 1)
        fused_value = fused_cdf(row_index, :);
        source_value = source_cdf(column_index, :);
        upper_covariance = upper_covariance + min(fused_value, source_value) - fused_value .* source_value;
        lower_covariance = lower_covariance + max(fused_value + source_value - 1, 0) - fused_value .* source_value;
    end
end

mixture = zeros(size(correlation));
positive_weight = positive & correlation ~= 0 & source_sd ~= 0 & fused_sd ~= 0;
negative_weight = negative & source_sd ~= 0 & fused_sd ~= 0;
mixture(positive_weight) = correlation(positive_weight) ./ ...
    (upper_covariance(positive_weight) ./ (fused_sd(positive_weight) .* source_sd(positive_weight)));
mixture(negative_weight) = correlation(negative_weight) ./ ...
    (lower_covariance(negative_weight) ./ (fused_sd(negative_weight) .* source_sd(negative_weight)));

joint_entropy = zeros(size(correlation));
for row_index = 1:size(source_pdf, 1)
    for column_index = 1:size(source_pdf, 1)
        upper = cdf_difference(fused_cdf, source_cdf, row_index, column_index, true);
        lower = cdf_difference(fused_cdf, source_cdf, row_index, column_index, false);
        bound = upper;
        bound(negative) = lower(negative);
        independent = fused_pdf(row_index, :) .* source_pdf(column_index, :);
        joint_pdf = mixture .* bound + (1 - mixture) .* independent;
        nonzero = joint_pdf ~= 0;
        entropy_term = zeros(size(joint_pdf));
        entropy_term(nonzero) = real(-joint_pdf(nonzero) .* log2(joint_pdf(nonzero)));
        joint_entropy = joint_entropy + entropy_term;
    end
end

source_entropy = entropy_from_pdf(source_pdf);
fused_entropy = entropy_from_pdf(fused_pdf);
mutual_information = source_entropy + fused_entropy - joint_entropy;
values = zeros(size(mutual_information));
nonzero_mi = mutual_information ~= 0;
values(nonzero_mi) = 2 * mutual_information(nonzero_mi) ./ ...
    (source_entropy(nonzero_mi) + fused_entropy(nonzero_mi));
end

function normalized = normalize_patches(patches)
minimum = min(patches, [], 1);
span = max(patches, [], 1) - minimum;
normalized = ones(size(patches));
nonconstant = span ~= 0;
normalized(:, nonconstant) = (patches(:, nonconstant) - minimum(nonconstant)) ./ span(nonconstant);
end

function result = cdf_difference(fused_cdf, source_cdf, row_index, column_index, upper)
current = cdf_bound(fused_cdf, source_cdf, row_index, column_index, upper);
previous_row = cdf_bound(fused_cdf, source_cdf, row_index - 1, column_index, upper);
previous_column = cdf_bound(fused_cdf, source_cdf, row_index, column_index - 1, upper);
previous_both = cdf_bound(fused_cdf, source_cdf, row_index - 1, column_index - 1, upper);
result = current - previous_row - previous_column + previous_both;
end

function result = cdf_bound(fused_cdf, source_cdf, row_index, column_index, upper)
if row_index == 0 || column_index == 0
    result = zeros(1, size(fused_cdf, 2));
    return;
end
fused_value = fused_cdf(row_index, :);
source_value = source_cdf(column_index, :);
if upper
    result = min(fused_value, source_value);
else
    result = max(fused_value + source_value - 1, 0);
end
end

function entropy = entropy_from_pdf(pdf)
term = zeros(size(pdf));
nonzero = pdf ~= 0;
term(nonzero) = -pdf(nonzero) .* log2(pdf(nonzero));
entropy = sum(term, 1);
end

function normalized = rerange_fast(image)
minimum = min(image(:));
maximum = max(image(:));
if maximum == minimum
    normalized = ones(size(image));
else
    normalized = (double(image) - minimum) / (maximum - minimum);
end
end
