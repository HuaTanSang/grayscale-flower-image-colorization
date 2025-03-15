from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image 

class FlowerDataset(Dataset): 
    def __init__(self, image_dir):
        self.image_dir = image_dir 

        self.rgb_transform = transforms.Compose ([
            transforms.Resize((128, 128)),
            transforms.ToTensor(), 
            #transforms.Normalize(mean=[0.3824, 0.3824, 0.3824], std=[0.2364, 0.236, 0.236])
        ])

        self.gray_scale_transform = transforms.Compose ([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.3824, 0.3824, 0.3824], std=[0.2364, 0.236, 0.236])
        ])

        self.__data = {} 
        for idx, image_path in enumerate(self.image_dir): 
            self.__data[idx] = image_path
        
    def __len__(self): 
        return len(self.__data)


    def __getitem__(self, index): 
        image_dir = self.__data[index]
        rgb_image = Image.open(image_dir)
        gray_scale = self.convert_to_grayscale(rgb_image)

        rgb_image = self.rgb_transform(rgb_image)
        gray_scale = self.gray_scale_transform(gray_scale)
        
        return {
            'gray_scale': gray_scale, 
            'rgb_scale': rgb_image
        }
    

    @staticmethod
    def convert_to_grayscale(image):
        gray_image = image.convert("L", resample=Image.NEAREST)
        gray_image = Image.merge("RGB", (gray_image, gray_image, gray_image))
        return gray_image 