# inference.py
import torch
from torchtext.data.utils import get_tokenizer
from models.complex_model import init_model
from dataset.dataset_loader import vocab, preprocess

# Path to the saved model. Update this to the desired checkpoint path
MODEL_PATH = "complex_model.pth"

# Load the saved model
model = init_model(len(vocab))
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Tokenizer
tokenizer = get_tokenizer("basic_english")

# Function to predict sentiment of a given text
def predict_sentiment(text):
    tokens = preprocess(text)
    tokens = torch.tensor(tokens, dtype=torch.int64).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(tokens)
        prediction = torch.sigmoid(output).item()
        sentiment = "positive" if prediction > 0.5 else "negative"
        return sentiment, prediction

if __name__ == "__main__":
    sample_text = "This is the best movie I have ever seen!"
    sentiment, prediction = predict_sentiment(sample_text)
    print(f"Sentiment prediction for '{sample_text}': {sentiment} ({prediction:.4f})")
