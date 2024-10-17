from flask import Flask, request, jsonify
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler

class LogisticRegression(nn.Module):
    def __init__(self, input_features=4, hidden_size=16):
        super(LogisticRegression, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Load the model
model = LogisticRegression()
model.load_state_dict(torch.load('model2.pkl', weights_only=True))
model.eval()

# Example training data to fit the scaler (replace with your actual training data)
X_train = np.array([
    [1, 4, 3, 4],
    [2, 3, 2, 1],
    [3, 2, 4, 2],
    # Add more samples as needed...
])

# Fit the scaler
scaler = StandardScaler()
scaler.fit(X_train)  # Fit the scaler on the training data

def predict_donation_probability(recency, frequency, monetary, time):
    # Create the input data with 4 original features
    input_data = np.array([[recency, frequency, monetary, time]])
    
    # Scale only the original features
    scaled_input = scaler.transform(input_data)  # This has shape (1, 4)
    
    with torch.no_grad():  # Disable gradient calculation
        scaled_input_tensor = torch.FloatTensor(scaled_input)  # Convert to tensor
        probability = model(scaled_input_tensor).item()  # Get the probability
    
    return probability

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello World"

@app.route('/predict', methods=['POST'])
def predict():
    # Get parameters from the request
    recency = float(request.args.get('recency', 1))  # Default to 1 if not provided
    frequency = float(request.args.get('frequency', 4))  # Default to 4 if not provided
    monetary = float(request.args.get('monetary', 3))  # Default to 3 if not provided
    time = float(request.args.get('time', 4))  # Default to 4 if not provided

    prob = predict_donation_probability(recency, frequency, monetary, time)

    return jsonify({'probability': prob})

if __name__ == "__main__":
    app.run(debug=True)
