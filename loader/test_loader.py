from pathlib import Path

from PIL import Image
from jittor import transform
from jittor.dataset import Dataset
from utils.RGB2YCrBb import RGB2YCrCb

to_tensor = transform.Compose([transform.ToTensor()])
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}

# from DePF: A Novel Fusion Approach based on Decomposition Pooling for Infrared and Visible Images
class TestLoader(Dataset):
    def __init__(self, data_dir, transform=to_tensor):
        super().__init__()
        data_dir = Path(data_dir)
        self.inf_path = data_dir / "ir"
        self.vis_path = data_dir / "vi"
        if not self.inf_path.is_dir() or not self.vis_path.is_dir():
            raise FileNotFoundError(
                f"Expected paired test directories: {self.inf_path} and {self.vis_path}"
            )
        infrared = sorted(
            path.name
            for path in self.inf_path.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        visible = sorted(
            path.name
            for path in self.vis_path.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not infrared:
            raise ValueError(f"No test images found in {self.inf_path}")
        if infrared != visible:
            raise ValueError("Infrared and visible test filenames do not match")
        self.name_list = infrared
        self.transform = transform
        self.set_attrs(total_len=len(self.name_list))

    def __getitem__(self, index):
        name = self.name_list[index]
        inf_image = Image.open(self.inf_path / name).convert('L')
        vis_image = Image.open(self.vis_path / name)
        image_size = inf_image.size
        inf_image = self.transform(inf_image)
        vis_image = self.transform(vis_image)
        vis_y_image, vis_cb_image, vis_cr_image = RGB2YCrCb(vis_image)
        return vis_image, vis_y_image, vis_cb_image, vis_cr_image, inf_image, name, image_size

    def __len__(self):
        return len(self.name_list)
