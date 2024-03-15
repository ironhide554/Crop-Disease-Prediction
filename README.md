# Crop-Disease-Prediction
This repository contains a deep learning model for predicting crop diseases using the ResNet-50 architecture. The model is trained on a dataset of images of various crop diseases to enable accurate classification.

# Requirements
Python (>=3.6)
TensorFlow (>=2.0)
Keras (>=2.3.1)
NumPy (>=1.16.0)
Matplotlib (>=3.0.0)
Installation
Clone this repository:

bash
Copy code
git clone https://github.com/ironhide554/Crop-Disease-Prediction.git
cd Crop-Disease-Prediction
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Usage
After installing the dependencies, place your dataset in the data/ directory. Ensure that the dataset is organized into separate folders, each corresponding to a different class (e.g., 'healthy', 'diseased').

# Train the model:

bash
Copy code
python train.py --data_dir data/ --epochs 50 --batch_size 32 --lr 0.001
Replace the values for --epochs, --batch_size, and --lr with your desired values.

# Evaluate the model:

bash
Copy code
python evaluate.py --model_path /path/to/saved_model.h5 --data_dir data/
Replace /path/to/saved_model.h5 with the path where your trained model is saved.

# Make predictions:

bash
Copy code
python predict.py --image_path /path/to/image.jpg --model_path /path/to/saved_model.h5
Replace /path/to/image.jpg with the path to the image you want to predict, and /path/to/saved_model.h5 with the path to your trained model.

# Model Architecture
The model architecture used in this project is ResNet-50, a deep convolutional neural network (CNN) architecture known for its effectiveness in image classification tasks. ResNet-50 consists of 50 layers, including convolutional layers, batch normalization, activation functions, and skip connections (residual connections), which help in mitigating the vanishing gradient problem during training.

# Dataset
The dataset used to train this model consists of images of various crop diseases and healthy crops. It is organized into separate folders, each containing images of a specific class. The dataset can be found at (https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset).

# License
This project is licensed under the MIT License. See the LICENSE file for details.

# Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs, feature requests, or questions.

# Contact
For any inquiries or assistance regarding the project, feel free to contact Sparsh Tiwari at sparshtiwari544@gmail.com.

Happy Coding!






 
