import pathlib
import cv2
import numpy as np
from jittor.dataset import Dataset



# from CHITNet: A Complementary to Harmonious Information Transfer Network for Infrared and Visible Image Fusion
class TrainLoader(Dataset):
    def __init__(
        self,
        ir_folder: pathlib.Path,
        vi_folder: pathlib.Path,
        patch_size=128,
        schedule_path=None,
    ):
        super(TrainLoader, self).__init__()
        self.ps = patch_size
        self.ir_list = [x for x in sorted(ir_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.vi_list = [x for x in sorted(vi_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        if not self.ir_list or not self.vi_list:
            raise ValueError(
                f"Training data is empty: ir={ir_folder}, vi={vi_folder}"
            )
        if len(self.ir_list) != len(self.vi_list):
            raise ValueError(
                f"Training pair count mismatch: ir={len(self.ir_list)}, "
                f"vi={len(self.vi_list)}"
            )
        for ir_path, vi_path in zip(self.ir_list, self.vi_list):
            if ir_path.name != vi_path.name:
                raise ValueError(
                    f"Training pair mismatch: ir={ir_path.name}, vi={vi_path.name}"
                )

        self.schedule_indices = None
        self.schedule_x = None
        self.schedule_y = None
        self.schedule_epoch = 0
        if schedule_path is not None:
            with np.load(str(schedule_path), allow_pickle=False) as schedule:
                self.schedule_indices = schedule["indices"].astype(np.int64, copy=True)
                self.schedule_x = schedule["crop_x"].astype(np.int64, copy=True)
                self.schedule_y = schedule["crop_y"].astype(np.int64, copy=True)
                if "filenames" in schedule:
                    expected = [str(name) for name in schedule["filenames"].tolist()]
                    actual = [path.name for path in self.ir_list]
                    if expected != actual:
                        raise ValueError("Training schedule filenames do not match the dataset")
            if not (
                self.schedule_indices.shape
                == self.schedule_x.shape
                == self.schedule_y.shape
            ):
                raise ValueError("Training schedule arrays must have identical shapes")
            if self.schedule_indices.ndim != 2:
                raise ValueError("Training schedule arrays must have shape [epochs, samples]")
            if self.schedule_indices.shape[1] != len(self.ir_list):
                raise ValueError(
                    "Training schedule sample count does not match the dataset"
                )
        self.set_attrs(total_len=len(self.ir_list))

    @property
    def scheduled(self):
        return self.schedule_indices is not None

    @property
    def schedule_epochs(self):
        return 0 if not self.scheduled else self.schedule_indices.shape[0]

    def set_epoch(self, epoch):
        if not self.scheduled:
            return
        if epoch < 0 or epoch >= self.schedule_epochs:
            raise ValueError(
                f"Schedule has {self.schedule_epochs} epochs, requested epoch {epoch}"
            )
        self.schedule_epoch = epoch

    def get_patch(self, ir, vis):
        H, W = ir.shape[1], ir.shape[2]
        x, y = np.random.randint(10, H-10-self.ps+1), np.random.randint(10, W-10-self.ps+1)
        ir_crop = ir[:, x:x+self.ps, y:y+self.ps]
        vis_crop = vis[:, x:x+self.ps, y:y+self.ps]
        return ir_crop, vis_crop
    
    def __getitem__(self, index):
        crop = None
        if self.scheduled:
            position = index
            index = int(self.schedule_indices[self.schedule_epoch, position])
            crop = (
                int(self.schedule_x[self.schedule_epoch, position]),
                int(self.schedule_y[self.schedule_epoch, position]),
            )
        ir_path = self.ir_list[index]
        vi_path = self.vi_list[index]
        assert ir_path.name == vi_path.name, f"Mismatch ir:{ir_path.name} vi:{vi_path.name}."
        ir = self.imread(path=ir_path, flags=cv2.IMREAD_GRAYSCALE)
        vi = self.imread(path=vi_path, flags=cv2.IMREAD_GRAYSCALE)
        if crop is None:
            ir_crop, vis_crop = self.get_patch(ir, vi)
        else:
            x, y = crop
            ir_crop = ir[:, x:x+self.ps, y:y+self.ps]
            vis_crop = vi[:, x:x+self.ps, y:y+self.ps]
            if ir_crop.shape[1:] != (self.ps, self.ps):
                raise ValueError(
                    f"Invalid scheduled crop for {ir_path.name}: x={x}, y={y}"
                )
        return ir_crop, vis_crop

    def __len__(self):
        return len(self.ir_list)

    @staticmethod
    def imread(path: pathlib.Path, flags=cv2.IMREAD_GRAYSCALE):
        im_cv = cv2.imread(str(path), flags)
        assert im_cv is not None, f"Image {str(path)} is invalid."
        im_ts = np.expand_dims(im_cv / 255., axis=0).astype(np.float32)
        return im_ts
