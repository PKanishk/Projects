SKIN DISEASE DETECTION

This project focuses on the detection of skin diseases using the EfficientNet architecture. The dataset used is HAM10000, which is available from the International Skin Imaging Collaboration (ISIC).

  Table of Contents
- [Introduction]
- [Dataset]
- [Prerequisites]
- [Installation]
- [Usage]
- [Training]
- [ENetModel Script]
- [Contributing]
- [License]


INTRODUCTION

Skin cancer is one of the most common types of cancer, and early detection is crucial for successful treatment. This project aims to leverage the power of deep learning and the EfficientNet architecture to classify different types of skin lesions using the HAM10000 dataset.


DATASET

The HAM10000 dataset, available from ISIC, contains 10,015 dermatoscopic images of pigmented lesions. These images are categorized into the following classes:
1. Melanocytic nevi (nv)
2. Melanoma (mel)
3. Benign keratosis-like lesions (bkl)
4. Basal cell carcinoma (bcc)
5. Actinic keratoses (akiec)
6. Vascular lesions (vasc)
7. Dermatofibroma (df)

You can download the dataset from [ISIC Archive](https://isic-archive.com/).


PREREQUISITES

- Python 3.7+
- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib


INSTALLATION

Clone the repository:
 bash:
  git clone https://github.com/yourusername/skin-disease-detection-efficientnet.git
  cd skin-disease-detection-efficientnet


Install the required packages:
 bash:
  pip install -r requirements.txt


USAGE

1. Download the Dataset:
   Download the HAM10000 dataset from the ISIC Archive and place it in the `data/` directory.

2. Preprocess the Data:
   Run the preprocessing script to prepare the data for training:
    bash:
     python EfficientNet.py

3. Train the Model:
   Start training the EfficientNet model:
    bash:
     python ENetModel.py


TRAINING

The training script (`EfficientNet.py`) trains the EfficientNet model on the HAM10000 dataset. You can configure the training parameters such as learning rate, batch size, and number of epochs in the script and evaluates the trained model on the test dataset and provides performance metrics such as accuracy.


ENETMODEL
The ENetModel(`ENetModel.py`) script provides the application for detecting skin diseases using the trained EfficientNet model.


CONTRIBUTION

Contributions are welcome! Please submit a pull request or open an issue to discuss any changes.


LICENSE

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


Feel free to modify this template to fit your specific project needs.
