# TextGuard: Hate Speech Classification System

## Overview

This project aims to build a deep learning model to classify speech as hateful or not hateful using a Long Short-Term Memory (LSTM) network. The model is trained on a dataset specifically curated for hate speech detection. Each time the training pipeline runs, it compares the performance of the newly trained model with the existing deployed model. If the new model outperforms the existing one and meets the performance threshold, the existing model is replaced for real-time predictions.

---

## Project Pipeline Diagram

### *End-to-End Workflow for Hate Speech Classification*
![alt text](<assets/Project Workflow.png>)


---

## Steps in the Project

### 1. Data Ingestion
- **Fetch Data from S3**: The dataset is fetched from an AWS S3 bucket, which contains labeled text data indicating whether the speech is hateful or not.

### 2. Data Validation
- **Check for Data Quality**: The raw data is inspected to ensure it meets the quality standards, including checks for missing values, invalid labels, or corrupted entries.

### 3. Data Transformation
- **Text Preprocessing**: This step includes tokenization, padding, and converting text into sequences suitable for feeding into the LSTM model.

### 4. Model Training (LSTM)
- **Train the LSTM Model**: The preprocessed text data is used to train an LSTM model that captures both short-term and long-term dependencies in the text to classify whether speech is hateful or not.
  
### 5. Model Evaluation
- **Evaluate Model Performance**: After training, the model's performance is assessed using metrics like accuracy.

### 6. Model Decision
- **Compare with Existing Model**: The newly trained model is compared against the currently deployed model. This is based on the predefined performance threshold. If the newly trained model exceeds this threshold and performs better than the existing model, it is selected for production.
  
### 7. Model Pusher
- **Model Replacement**: If the new model passes the comparison check, it replaces the existing model in the production environment and is deployed for real-time predictions.

### 8. Dockerization
- **Create Docker Image**: The selected model is Dockerized to ensure consistency across different environments.

### 9. CI/CD Pipeline
- **CircleCI Pipeline**: A CI/CD pipeline is used to automate testing and deployment. The Docker image is pushed to AWS Elastic Container Registry (ECR), and the model is deployed to an EC2 instance.

### 10. Real-Time Prediction
- **FastAPI Prediction Pipeline**: The deployed model is integrated with a FastAPI server to allow for real-time predictions based on input text. The system receives text input, runs it through the trained model, and returns the prediction: "Hateful" or "Not Hateful."

---

## Installation

### Prerequisites
- Python 3.x
- Docker
- AWS CLI
- CircleCI account
- FastAPI framework

### Steps

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/chaitanya-24/Hate_Speech.git
    cd Hate_Speech
    ```

2. **Install Required Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Configure AWS CLI**:
    ```bash
    aws configure
    ```
    *Follow the aws cli configuration steps like setting up IAM user access key id and access key, etc*
---

## Usage

1. **Run Model Training Pipeline**:
    ```bash
    python pipeline/train_pipeline.py
    ```
    - This will train the LSTM model on the dataset and automatically evaluate its performance.
    - The system will compare the performance of the newly trained model with the current model and decide whether to replace the existing model.

2. **Run FastAPI Server for Predictions**:
    ```bash
    python app.py
    ```

---

## Threshold-Based Model Selection

Each time the model training pipeline is executed, the following comparison steps occur:
1. **Evaluate Performance**: The newly trained model's accuracy is calculated.
2. **Compare with Existing Model**: If the accuracy of the new model is greater than both the threshold and the performance of the current model, the new model replaces the old model in production.
3. **Update Production**: The new model is deployed using Docker and pushed to an EC2 instance via CircleCI.

---

## Technologies Used

- **Programming Language**: Python
- **Deep Learning Framework**: TensorFlow / Keras (LSTM)
- **Web Framework**: FastAPI
- **Cloud Services**: AWS S3, AWS EC2, AWS ECR
- **Containerization**: Docker
- **CI/CD**: CircleCI
- **Model Deployment**: FastAPI, Docker, CircleCI

---

## Author
### Chaitanya Sawant:
* **[Gmail](mailto:chaitanya.aiwork@gmail.com)**
* **[Twitter](https://twitter.com/chaitanya_42)**

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

