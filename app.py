from flask import Flask, request, jsonify, send_from_directory, render_template  # type: ignore
from flask_cors import CORS
import torch
from cvnl_intent_rnn import IntentClassifierRNN, EmbeddingPackable
import json
import os

app = Flask(__name__)
CORS(app)

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

# Load intent model and data
intent_checkpoint = torch.load('intent_model.pth')
intent_config = intent_checkpoint['config']
intent_model = IntentClassifierRNN(
    vocab_size=intent_config['vocab_size'],
    embed_dim=intent_config['embed_dim'],
    hidden_dim=intent_config['hidden_dim'],
    output_dim=intent_config['output_dim'],
    num_layers=intent_config['num_layers'],
    bidirectional=intent_config['bidirectional']
)
intent_model.load_state_dict(intent_checkpoint['model_state_dict'])
intent_model.eval()

# Load vocabularies and mappings
intent_vocab = intent_checkpoint['vocab']
intent_idx_to_label = intent_checkpoint['idx_to_label']

def get_intent_description(intent):
    descriptions = {
        'make_call': 'Making a phone call or voice communication',
        'search_song': 'Searching for music or songs',
        'play_music': 'Playing music or audio content',
        'get_weather': 'Checking weather conditions',
        # Add more descriptions as needed
    }
    return descriptions.get(intent, 'General user request')

@app.route('/analyze-intent', methods=['POST'])
def analyze_intent():
    data = request.json
    text = data.get('text', '')
    
    # Preprocess text (using the same preprocessing as in training)
    tokens = text.lower().split()
    indices = [intent_vocab.get(token, intent_vocab['<unk>']) for token in tokens]
    input_tensor = torch.tensor([indices])
    
    # Get prediction and top 3 intents
    with torch.no_grad():
        outputs = intent_model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        
        # Get top 3 predictions
        top_probs, top_indices = torch.topk(probs[0], k=min(3, len(intent_idx_to_label)))
        
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



if __name__ == '__main__':
    app.run(port=5000)