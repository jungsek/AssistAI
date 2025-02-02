from flask import Flask, request, jsonify, send_from_directory, render_template  # type: ignore
from flask_cors import CORS
import torch
import torch.nn as nn
from models.cvnl_asl_cnn import imageProcessor, cnnModel
from models.cvnl_emotions_rnn import LastTimeStep, EmbeddingPackable, better_tokenizer
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import json
import os
import cv2
import numpy as np
import base64

app = Flask(__name__)
CORS(app)

# Classes for RNN
class LastTimeStep(nn.Module):
    def __init__(self, rnn_layers=1, bidirectional=False):
        super(LastTimeStep, self).__init__()
        self.rnn_layers = rnn_layers
        self.num_driections = 2 if bidirectional else 1

    def forward(self, input):
        rnn_output = input[0]
        last_step = input[1]
        if(type(last_step) == tuple):
            last_step = last_step[0]
        batch_size = last_step.shape[1]
        last_step = last_step.view(self.rnn_layers, self.num_driections, batch_size, -1)
        last_step = last_step[self.rnn_layers-1]
        last_step = last_step.permute(1, 0, 2)
        return last_step.reshape(batch_size, -1)

class EmbeddingPackable(nn.Module):
    def __init__(self, embd_layer):
        super(EmbeddingPackable, self).__init__()
        self.embd_layer = embd_layer

    def forward(self, input):
        if type(input) == torch.nn.utils.rnn.PackedSequence:
            sequences, lengths = torch.nn.utils.rnn.pad_packed_sequence(input.cpu(), batch_first=True)
            sequences = self.embd_layer(sequences.to(input.data.device))
            return torch.nn.utils.rnn.pack_padded_sequence(sequences, lengths.cpu(),
                                                         batch_first=True, enforce_sorted=False)
        else:
            return self.embd_layer(input)


# Set up Kaggle credentials
os.environ['KAGGLE_USERNAME'] = 'jungsek'  
os.environ['KAGGLE_KEY'] = 'bc2f4055959e862034c06865963fb548'    

# Add route to serve the main page
@app.route('/')
def serve_index():
    return render_template('index.html')

# Add route to serve static files
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

# Load the model and configurations
intent_checkpoint = torch.load('./models/intent_model.pth', map_location=torch.device('cpu'))
vocab_size = len(intent_checkpoint['vocab'])
D = 80
hidden_nodes = 112
num_layers = 2
bidirectional = True
dropout = 0.5
num_classes = len(intent_checkpoint['label_to_idx'])

# Create the model
rnn_model = nn.Sequential(
    EmbeddingPackable(nn.Embedding(vocab_size, D)),
    nn.GRU(
        input_size=D,
        hidden_size=hidden_nodes,
        batch_first=True,
        bidirectional=bidirectional,
        num_layers=num_layers,
        dropout=dropout
    ),
    LastTimeStep(rnn_layers=num_layers, bidirectional=bidirectional),
    nn.LayerNorm(hidden_nodes * 2),
    nn.Dropout(p=0.7),
    nn.Linear(hidden_nodes * 2, num_classes)
)

# Load the trained weights
rnn_model.load_state_dict(intent_checkpoint['model_state_dict'])
rnn_model.eval()

# Load vocabularies and mappings
intent_vocab = intent_checkpoint['vocab']
intent_idx_to_label = intent_checkpoint['idx_to_label']

def get_intent_description(intent):
    descriptions = {
        'make_call': 'Making a phone call or voice communication',
        'search_song': 'Searching for music or songs',
        'play_music': 'Playing music or audio content',
        'get_weather': 'Checking weather conditions',
        'restaurant_reservation': 'Making a restaurant reservation',
        'share_location': 'Sharing current location',
        'greeting': 'General greeting or hello',
        'goodbye': 'Saying goodbye or ending conversation',
        'directions': 'Getting directions to a location',
        'traffic': 'Checking traffic information',
        # Add more descriptions as needed
    }
    return descriptions.get(intent, 'General user request')

@app.route('/analyze-intent', methods=['POST'])
def analyze_intent():
    data = request.json
    text = data.get('text', '')
    
    # Preprocess text
    tokens = text.lower().split()
    indices = torch.tensor([[intent_vocab.get(token, intent_vocab['<unk>']) for token in tokens]])
    
    # Get prediction
    with torch.no_grad():
        outputs = rnn_model(indices)
        probs = torch.softmax(outputs, dim=1)
        
        # Get top 3 predictions
        top_probs, top_indices = torch.topk(probs[0], k=min(3, num_classes))
        
        # Format results
        top_intents = [ 
            {
                'intent': intent_idx_to_label[idx.item()],
                'confidence': prob.item(),
                'description': get_intent_description(intent_idx_to_label[idx.item()])
            }
            for idx, prob in zip(top_indices, top_probs)
        ]
    
    return jsonify({
        'top_intents': top_intents,
        'input_text': text
    })

emotion_checkpoint = torch.load('./models/rnn_emotions.pth', map_location=torch.device('cpu'), weights_only=False)
vocab_size = len(emotion_checkpoint['vocab'])
D = 126
hidden_nodes = 64
num_layers = 4
bidirectional = True
dropout = 0.5
classes = len(emotion_checkpoint['new_labels_order'])

emotion_model = nn.Sequential(
  EmbeddingPackable(nn.Embedding(vocab_size, D)),
  nn.GRU(
      input_size=D,
      hidden_size=hidden_nodes,
      num_layers=num_layers,
      batch_first=True,
      dropout=dropout if num_layers > 1 else 0,
      bidirectional=bidirectional,
  ),
  LastTimeStep(rnn_layers=num_layers, bidirectional=bidirectional),
  nn.LayerNorm(hidden_nodes * (2 if bidirectional else 1)),
  nn.Dropout(0.5),
  nn.Linear(
      in_features=hidden_nodes * (2 if bidirectional else 1),
      out_features=classes),
)

# Load the trained weights
emotion_model.load_state_dict(emotion_checkpoint['model_state_dict'])
emotion_model.eval()

# Load vocabularies and mappings
emotion_vocab = emotion_checkpoint['vocab']
emotion_idx_to_label = emotion_checkpoint['new_labels_order']

@app.route('/analyze-emotion', methods=['POST'])
def analyze_emotion():
    data = request.json
    text = data.get('text', '')
    
    # Preprocess text
    tokens = better_tokenizer(text)
    token_ids = [emotion_vocab.get(tok, 1) for tok in tokens]  # 1 = <unk>
    
    # Convert to tensor and pack sequence
    input_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(1)  # (seq_len, 1)
    lengths = torch.tensor([len(token_ids)])

    # Create packed sequence matching model's expected format
    packed = pack_padded_sequence(
        input_tensor,
        lengths,
        batch_first=False,  # Must match model's GRU configuration!
        enforce_sorted=False
    )

    # Get prediction
    with torch.no_grad():
        outputs = emotion_model(packed)
        probs = torch.sigmoid(outputs).cpu()
        
    # Extract confidence scores
    threshold = 0.3  # Define a confidence threshold
    confidences = {
        emotion_checkpoint['new_labels_order'][i]: float(probs[0][i])
        for i in range(len(emotion_checkpoint['new_labels_order']))
        if probs[0][i] > threshold
    }

    # If no emotion meets threshold, return most likely one
    if not confidences:
        most_likely_label = emotion_checkpoint['new_labels_order'][torch.argmax(probs).item()]
        confidences = {most_likely_label: 1.0}  # Assign 100% confidence to most likely label
    
    formatted_emotions = [
        {
            "emotion": emotion,
            "confidence": round(confidence, 3)  # Keep confidence to 3 decimal places
        }
        for emotion, confidence in confidences.items()
    ]

    return jsonify({
        "top_emotions": formatted_emotions,
        "input_text": text
    })

#load CNN stuff
imgProcessor = imageProcessor()
cnn = cnnModel()
cnn.loadWeights("./models/cnn_model_3.pth")

@app.route('/analyze-cnn', methods=['POST'])
def analyze_asl():
    data = request.json
    imgData = data['image']
    if ',' in data['image']:
        imgData = imgData.split(',')[1]
    
    #decode and convert to cv2
    bytes = base64.b64decode(imgData)
    npArr = np.frombuffer(bytes, np.uint8)
    if not npArr.size: return jsonify("No image provided")

    img = cv2.imdecode(npArr, cv2.IMREAD_COLOR)

    #input into model
    img = imgProcessor.processImage(img)
    probabilities = sorted(cnn.predictImage(img), key=lambda x: x[1], reverse=True)
    print(probabilities)
    result = probabilities[0][0]
    return jsonify(result)

if __name__ == '__main__':
    app.run(port=5000, debug=True)