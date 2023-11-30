import io
import torch
import pandas as pd
from PIL import Image
import torch.utils.data as data
from torchvision import transforms
from typing import Dict, List, Tuple, Callable


class DeepFashionMaskDataset(data.Dataset):
    def __init__(
        self,
        root_path: str,
        is_train: bool,
        target_size_hw: Tuple[int, int],
        transform: Callable,
        data_keys=["images", "mask"],
    ) -> None:
        self.is_train = is_train
        self.target_size_hw = target_size_hw
        self.transform = transform
        self.root_path = root_path
        self.data = self._load_data(self.root_path, data_keys)

    def _load_data(self, root_path: str, data_keys: List[str]) -> pd.DataFrame:
        # TODO: read data from all files in root_path
        df = pd.read_parquet(root_path, engine="pyarrow")
        return df[data_keys]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        image_dict = self.data["images"][index]
        mask_dict = self.data["mask"][index]

        byte_image = image_dict["bytes"]
        byte_mask = mask_dict["bytes"]

        image = Image.open(io.BytesIO(byte_image)).convert("RGB")
        mask = Image.open(io.BytesIO(byte_mask))

        mask_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(self.target_size_hw, antialias=True),
            ]
        )

        mask_tensor = mask_transform(mask)
        image_tensor = self.transform(image)

        return {"image": image_tensor.float(), "mask": mask_tensor}
