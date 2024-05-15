from flask import Flask, request, jsonify
from PIL import Image
from sklearn.preprocessing import StandardScaler
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
best_model = load_model('best_emnist_model.h5')

# Define the alpha_num_to_char dictionary
alpha_num_to_char = { 0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f', 16: 'g', 17: 'h', 18: 'i', 19: 'j', 20: 'k', 21: 'l', 22: 'm', 23: 'n', 24: 'o', 25: 'p', 26: 'q', 27: 'r', 28: 's', 29: 't', 30: 'u', 31: 'v', 32: 'w', 33: 'x', 34: 'y', 35: 'z', 36: 'A', 37: 'B', 38: 'C', 39: 'D', 40: 'E', 41: 'F', 42: 'G', 43: 'H', 44: 'I', 45: 'J', 46: 'K', 47: 'L', 48: 'M', 49: 'N', 50: 'O', 51: 'P', 52: 'Q', 53: 'R', 54: 'S', 55: 'T', 56: 'U', 57: 'V', 58: 'W', 59: 'X', 60: 'Y', 61: 'Z' }

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the request
    image = request.files['image']

    # Open and preprocess the image
    image = Image.open(image)
    image = image.resize((28, 28))
    image = image.convert('L')
    scaler = StandardScaler()
    image_array = np.array(image)
    image_array = scaler.fit_transform(image_array.reshape(-1, 1)).reshape(28, 28)
    image_array = image_array.reshape(1, 28, 28, 1)

    # Make predictions on the image
    predictions = best_model.predict(image_array)

    # Get the predicted digit (index with highest probability)
    predicted_digit = np.argmax(predictions)
    predicted_letter = alpha_num_to_char[predicted_digit]

    # Return the predicted letter
    return jsonify({'predicted_letter': predicted_letter})

if __name__ == '__main__':
    app.run(debug=True)

