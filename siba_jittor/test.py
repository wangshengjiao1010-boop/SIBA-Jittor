import os
from jittor import transform
from tqdm import tqdm
from loader.test_loader import TestLoader
from utils.RGB2YCrBb import YCrCb2RGB, clamp
from models.SIBA import SIBA
import jittor as jt
import time

jt.flags.use_cuda = 1

model_path = "checkpoint/SIBA_epoch60.pkl"
testdata_path = '/home/ws/datasets/image_fusion/MSRS'
result_save_path = '/home/ws/datasets/image_fusion/MSRS_fusion_result'

if not os.path.exists(result_save_path):
    os.makedirs(result_save_path)

model = SIBA()
model.load_parameters(jt.load(model_path)['model'])

total = sum([params.numel() for params in model.parameters()])
print("Number of params: {%.3f M}" % (total / 1e6))
model.eval()


test_dataset = TestLoader(testdata_path)
test_loader = test_dataset.set_attrs(batch_size=1, shuffle=False, num_workers=1, drop_last=False)

sum_time = 0
with jt.no_grad():
    for _, vis_y_image, cb, cr, ir_image, img_name, _ in tqdm(test_loader, total=test_loader.__batch_len__()):

        start = time.time()
        image_fused = model(ir_image,vis_y_image)
        end = time.time()

        sum_time+=(end-start)
        
        image_fused = clamp(image_fused[0])
        image_fused = YCrCb2RGB(image_fused, cb[0], cr[0])
        image_fused = transform.ToPILImage()(image_fused.transpose(1, 2, 0))
        image_fused.save(f'{result_save_path}/{img_name[0]}')

print('use time: ', sum_time)
