# INM-706 Courswork - Sentiment Analysis with Deep Learning

## Project Structure
- `dataset/`: Contains the dataset loader script.
  - `dataset_loader.py`: Script to load and preprocess the IMDB dataset.
- `models/`: Contains the model definitions.
  - `baseline_model.py`: Baseline model definition.
  - `complex_model.py`: Complex model definition with BiLSTM and attention mechanism.
- `train_baseline_model.py`: Script to train and evaluate the baseline model.
- `train_complex_model.py`: Script to train and evaluate the complex model.
- `inference.py`: Script to run inference using the trained complex model.
- `requirements.txt`: List of dependencies.
- `setup.sh`: Script to set up the environment.

## Instructions

### Setup Environment
1. Run `setup.sh` to create a virtual environment and install dependencies:
   ```bash
   bash setup.sh
   source venv/bin/activate

### Train Models
1. Train the baseline model:
    ```bash
    python train_baseline_model.py

2. Train the complex model:
    ```bash
    python train_complex_model.py

### Run Inference
1. Run the inference script to predict the sentiment of a new text:
    ```bash
    python inference.py

2. To predict the sentiment of custom text, modify the `sample_text` variable in the `inference.py` script:
    ```bash
    if __name__ == "__main__":
        sample_text = "Your custom text here"
        sentiment, prediction = predict_sentiment(sample_text)
        print(f"Sentiment prediction for '{sample_text}': {sentiment} ({prediction:.4f})")

3. To use a different model, modify the `MODEL_PATH` variable:
    ```bash
    MODEL_PATH = "complex_model.pth"



