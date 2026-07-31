function [overall_quality, quality, quality_map] = analysis_ms_ssim_fast(imgSeq, fI, K, window, level, weight)
if nargin < 2 || nargin > 6
    overall_quality = -Inf;
    quality = -Inf;
    quality_map = -Inf;
    return;
end
if ~exist('K', 'var')
    K = 0.03;
end
if ~exist('window', 'var')
    window = fspecial('gaussian', 11, 1.5);
end
if ~exist('level', 'var')
    level = 3;
end
if ~exist('weight', 'var')
    weight = [0.0448 0.2856 0.3001]';
    weight = weight / sum(weight);
end
if level ~= length(weight)
    overall_quality = -Inf;
    quality = -Inf;
    quality_map = -Inf;
    return;
end

[height, width, source_count] = size(imgSeq);
minimum_width = min(height, width) / (2^(level - 1));
if minimum_width < max(size(window))
    overall_quality = -Inf;
    quality = -Inf;
    quality_map = Inf;
    return;
end

imgSeq = double(imgSeq);
fI = double(fI);
downsample_filter = ones(2) / 4;
quality = zeros(level, 1);
quality_map = cell(level, 1);
if level == 1
    [quality, quality_map] = mef_ssim_fast(imgSeq, fI, K, window);
    overall_quality = quality;
    return;
end

for scale = 1:level - 1
    [quality(scale), quality_map{scale}] = mef_ssim_fast(imgSeq, fI, K, window);
    source_copy = imgSeq;
    clear imgSeq;
    for source_index = 1:source_count
        source = squeeze(source_copy(:, :, source_index));
        downsampled = imfilter(source, downsample_filter, 'symmetric', 'same');
        imgSeq(:, :, source_index) = downsampled(1:2:end, 1:2:end);
    end
    downsampled = imfilter(fI, downsample_filter, 'symmetric', 'same');
    clear fI;
    fI = downsampled(1:2:end, 1:2:end);
end
[quality(level), quality_map{level}] = mef_ssim_fast(imgSeq, fI, K, window);
quality = quality(:);
overall_quality = prod(quality.^weight);
end
