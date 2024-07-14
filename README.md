# Real-Time Muscle Stress Prediction for Shoulder Supported Exoskeleton Interaction

## Introduction
The primary objective of this project is to develop an application that utilizes a deep learning model to accurately predict real-time stress caused in a muscle. This application will be seamlessly integrated into a digital platform specifically crafted to evaluate the performance of shoulder-supported exoskeleton interaction. By leveraging advanced deep learning techniques, the system aims to provide precise and real-time insights into muscle stress, which is crucial for optimizing the performance and safety of exoskeletons used in various industrial and medical applications.

## Project Overview
The project involves the following key components:

* **Data Collection**: Gathering muscle stress data using sensors and integrating it into a structured dataset.
* **Data Preprocessing**: Cleaning and preparing the data for training by normalizing features and segmenting time-series data.
* **Model Development**: Designing and training a deep learning model (LSTM) to predict muscle stress based on the preprocessed data.
* **Real-Time Prediction**: Implementing the trained model in a real-time environment to provide continuous stress predictions.
* **Platform Integration**: Integrating the predictive model into a digital platform for real-time monitoring and performance evaluation of shoulder-supported exoskeletons.

## Table of Contents
1.  [Introduction](#introduction)
2.  [Overview](#Overview)
3.  [Scope](#Scope)
4.  [Objectives](#Objectives)
5.  [Installation](#Installation)
6.  [Usage](#Usage)
7.  [Features](#Features)
8.  [Testing](#Testing)
9.  [Contributing](#Contributing)
10. [License](#License)
11. [Acknowledgements](#Acknowledgements)

## Scope
The scope of this project includes the development and integration of a real-time muscle stress prediction model into a comprehensive digital platform designed for exoskeleton performance evaluation.

## Key Objectives
* **Accurate Stress Prediction:** Develop a deep learning model capable of accurately predicting muscle stress in real-time.
* **Real-Time Integration:** Ensure the model can operate in a real-time environment for continuous monitoring.
* **Platform Development:** Integrate the model into a digital platform to assess exoskeleton performance.
* **User-Friendly Interface:** Create an intuitive interface for easy interaction with the prediction system.


## Installation

### Steps
Clone the repository: 
```
git clone git@github.com:Vikneshwara-kumar/DL-Based-Muscle-Strain-Prediction-System.git
```

Navigate to the project directory: 
```
cd DL-Based-Muscle-Strain-Prediction-System
```

To set up the environment and install dependencies, use the provided setup_environment.sh script:
```
chmod +x setup_environment.sh
./setup_environment.sh
```

## Usage
1.  **Prepare the Data:** Ensure your dataset is available in the Dataset/Train.csv file.
2.  **Run the Main Script:** Execute the main script to start the training process.
```
python main.py
```
3.  **Results:** After the training is complete, the following files will be saved in the results directory:
*   training_validation_loss.png: Plot of training and validation loss over epochs.
*   confusion_matrix.png: Heatmap of the confusion matrix.
*   lstm_model.keras: Trained LSTM model.
*   scaler.pkl: Scaler used for data preprocessing.

## Features
*   Data Preprocessing: Scales and segments time-series data for model training.
*   LSTM Model: Deep learning model for predicting muscle stress.
*   Training and Evaluation: Includes training the model and evaluating its performance on test data.
*   Result Visualization: Generates plots for training loss and confusion matrix.

## Testing
To test the individual components:

1.  Unit Tests: Write unit tests for each function to ensure they work correctly.
2.  Integration Tests: Test the integration of the preprocessing steps and model training.

##  Contributing
Contributions are welcome! Please follow these steps:

1.  Fork the repository.
2.  Create your feature branch (git checkout -b feature/YourFeature).
3.  Commit your changes (git commit -m 'Add some feature').
4.  Push to the branch (git push origin feature/YourFeature).
5.  Open a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

##  Acknowledgements
Special thanks to the research community for providing insights and resources.