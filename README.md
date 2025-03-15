# Flower Colorization using UNet++

## 🌸 Introduction
This project focuses on colorizing grayscale flower images using UNet++, a powerful deep learning model for image-to-image translation. The model is trained to learn the mapping from grayscale to RGB images, enhancing the visual appeal of flower photographs.

## 📌 Features
- **Deep Learning-based Image Colorization**: Utilizes UNet++ architecture to predict colorized images from grayscale inputs.
- **High-Quality Colorization**: Produces natural and vivid colors for flower images.
- **Supports Custom Datasets**: Easily adaptable to different flower image datasets.
- **PyTorch-based Implementation**: Built using PyTorch for flexibility and efficiency.

## 📂 Dataset
The dataset consists of:
- **Grayscale Images**: Input images converted to grayscale.
- **RGB Ground Truth**: The original colored images used as the reference for training.


## 🏗️ Model Architecture
The project employs **UNet++**, an advanced version of the original UNet model with nested and dense skip connections, improving the model’s ability to learn fine-grained details.

## 🚀 Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/HuaTanSang/flower-colorization-using-unet-
   cd flower-colorization-using-unet-
   ```
2. Ensure that the dataset is correctly placed in the `dataset/` directory.

## 🏋️‍♂️ Training
Run the following command to train the model:
```sh
python main.py
```
The training process includes:
- Using **L1 Loss** as the loss function.
- Evaluating performance using **PSNR** and **SSIM** metrics.
- Implementing early stopping based on SSIM scores.
- Saving model checkpoints during training.

## 🔍 Inference
To perform colorization on a grayscale image:
```sh
python inference.py --input path/to/grayscale/image.png --output path/to/save/colorized/image.png
```

## 📊 Evaluation
The model is evaluated using:
- **PSNR (Peak Signal-to-Noise Ratio)**
- **SSIM (Structural Similarity Index Measure)**
- **L1/L2 Loss**


## 🔧 Troubleshooting
- **ValueError: Axes don’t match array**: Ensure that the image tensor is transposed correctly using `.transpose(1, 2, 0)` for displaying.
- **CUDA Tensor to NumPy Conversion Error**: Convert tensors to CPU before using NumPy with `.cpu().numpy()`.
- **Dataset Loading Issue**: Ensure that the dataset path is correct and properly structured.

## 📜 License
This project is licensed under the MIT License.

