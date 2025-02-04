from PIL import Image, ImageFilter, ImageDraw
import cv2
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
import random


class CartoonDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        target_size: int = 1024,
        image_size: int = 1024,
        padding: int = 0,
        model_type: str = "cartoon",
        drop_text_prob: float = 0.1,
        drop_image_prob: float = 0.1,
        return_pil_image: bool = False,
    ):
        self.base_dataset = base_dataset
        self.target_size = target_size
        self.image_size = image_size
        self.padding = padding
        self.model_type = model_type
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.return_pil_image = return_pil_image

        self.to_tensor = T.ToTensor()


    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data = self.base_dataset[idx]
        condition_img = data['condition']
        target_image = data['target']

        # Tag
        tag = data['tags'][0]

        description = {
            "lion": "lion like animal",
            "bear": "bear like animal",
            "gorilla": "gorilla like animal",
            "dog": "dog like animal",
            "elephant": "elephant like animal",
            "eagle": "eagle like bird",
            "tiger": "tiger like animal",
            "owl": "owl like bird",
            "woman": "woman",
            "parrot": "parrot like bird",
            "mouse": "mouse like animal",
            "man": "man",
            "pigeon": "pigeon like bird",
            "girl": "girl",
            "panda": "panda like animal",
            "crocodile": "crocodile like animal",
            "rabbit": "rabbit like animal",
            "boy": "boy",
            "monkey": "monkey like animal",
            "cat": "cat like animal"
        }

        # Resize the image
        condition_img = condition_img.resize((self.target_size, self.target_size)).convert("RGB")
        target_image = target_image.resize((self.target_size, self.target_size)).convert("RGB")

        blank = Image.new("RGB", (2*self.target_size + 10, self.target_size), (255, 255, 255))

        blank.paste(condition_img, (0, 0))
        blank.paste(target_image, (self.target_size + 10, 0))

        target_image = blank

        # Process datum to create description
        description = f"layout - side by side photos of a {description[tag]} cartoon character in a white background. Left: a cartoon; Right: a cartoon."

        # Randomly drop text or image
        drop_text = random.random() < self.drop_text_prob
        drop_image = random.random() < self.drop_image_prob
        if drop_text:
            description = ""
        if drop_image:
            condition_img = Image.new(
                "RGB", (self.condition_size, self.condition_size), (0, 0, 0)
            )


        return {
            "image": self.to_tensor(target_image),
            # "condition": self.to_tensor(condition_img),
            "model_type": self.model_type,
            "description": description,
            # 16 is the downscale factor of the image
            # "position_delta": np.array([0, -16]),
        }



class RoomDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        target_size: int = 1024,
        image_size: int = 1024,
        padding: int = 0,
        model_type: str = "cartoon",
        drop_text_prob: float = 0.1,
        drop_image_prob: float = 0.1,
        return_pil_image: bool = False,
    ):
        self.base_dataset = base_dataset
        self.target_size = target_size
        self.image_size = image_size
        self.padding = padding
        self.model_type = model_type
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.return_pil_image = return_pil_image

        self.to_tensor = T.ToTensor()


    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data = self.base_dataset[idx]
        image1 = data['image1']
        image2 = data['image2']
        image1_description = data['image1_description']
        image2_description = data['image2_description']



        # Resize the image
        image1 = image1.resize((self.target_size, self.target_size)).convert("RGB")
        image2 = image2.resize((self.target_size, self.target_size)).convert("RGB")

        blank = Image.new("RGB", (2*self.target_size + 10, self.target_size), (255, 255, 255))

        blank.paste(image1, (0, 0))
        blank.paste(image2, (self.target_size + 10, 0))

        target_image = blank

        # Process datum to create description
        description = f"Layout- two image of a room side by side. Left: {image1_description}; Right: {image2_description};"

        # Randomly drop text or image
        drop_text = random.random() < self.drop_text_prob
        drop_image = random.random() < self.drop_image_prob
        if drop_text:
            description = ""
        if drop_image:
            image1 = Image.new(
                "RGB", (self.condition_size, self.condition_size), (0, 0, 0)
            )


        return {
            "image": self.to_tensor(target_image),
            "model_type": self.model_type,
            "description": description,
        }

