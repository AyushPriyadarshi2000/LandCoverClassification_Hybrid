# LandCoverClassification_Hybrid

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Dataset](#dataset)
4. [Model Architecture](#model-architecture)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Usage](#usage)
8. [Results](#results)
9. [Contributing](#contributing)
10. [License](#license)

## Project Overview
LandCoverClassification_Hybrid is a hybrid model designed for multi-label land cover classification. The model integrates a Convolutional Neural Network (CNN) backbone (ResNet34) with a Recurrent Neural Network (RNN) module (LSTM) to capture spatial and temporal dependencies in land cover images. The goal is to improve the accuracy of land cover classification by leveraging the strengths of both CNNs and RNNs.

## Installation
To get started with LandCoverClassification_Hybrid, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/LandCoverClassification_Hybrid.git
   cd LandCoverClassification_Hybrid
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset
The dataset used for this project consists of land cover images with corresponding labels. You can download the dataset from [this link](#). Make sure to place the dataset in the `data/` directory.

## Model Architecture
The hybrid model consists of three main components:
1. **CNN Backbone (ResNet34)**: Extracts spatial features from the images.
2. **RNN Module (LSTM)**: Captures temporal dependencies in the extracted features.
3. **Output Layer**: Produces the final multi-label classification output.

## Training
To train the model, use the following command:
```bash
python train.py --data_dir data/ --epochs 50 --batch_size 32 --learning_rate 0.001
```
This command will train the model using the dataset in the `data/` directory for 50 epochs with a batch size of 32 and a learning rate of 0.001.

### Training Techniques
- **Learning Rate Finder**: Automatically finds the optimal learning rate.
- **One-Cycle Policy**: Optimizes the learning rate schedule for better performance.

## Evaluation
To evaluate the trained model, use the following command:
```bash
python evaluate.py --model_path models/best_model.pth --data_dir data/
```
This command will evaluate the model using the test dataset and print the classification metrics.

## Usage
After training, you can use the model to classify new land cover images. Use the following command to classify a single image:
```bash
python classify.py --model_path models/best_model.pth --image_path images/sample.jpg
```
This command will output the predicted labels for the input image.

## Results
The results of the model, including accuracy, precision, recall, and F1-score, are presented in the `results/` directory. The model achieves state-of-the-art performance on the provided dataset.

## Contributing
We welcome contributions to LandCoverClassification_Hybrid! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.

### Steps to Contribute
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes and push to your branch.
4. Open a pull request to the main branch.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
