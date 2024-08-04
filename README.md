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
