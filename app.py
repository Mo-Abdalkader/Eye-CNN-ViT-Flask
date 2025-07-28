import os
from flask import Flask, render_template, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# Import your model architectures
from models.architecture import DualOutputResNet50, DualOutputViT

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def load_model_with_state_dict(path, model_class, device):
    """Load model by instantiating architecture and loading state dict"""
    try:
        # First try to load as complete model
        model = torch.load(path, map_location=device)
        if hasattr(model, 'eval'):
            model.eval()
            print(f"✅ Loaded complete model from {path}")
            return model
    except:
        pass

    # If that fails, load as state dict
    try:
        # Instantiate the model architecture with pretrained=False since we're loading weights
        model = model_class(pretrained=False)

        # Load the state dictionary
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        print(f"✅ Loaded model state dict from {path}")
        return model
    except Exception as e:
        print(f"❌ Error loading model from {path}: {e}")
        raise


device = torch.device("cpu")

# Load CNN model (ResNet50)
try:
    cnn_model = load_model_with_state_dict("models/cnn_model.pth", DualOutputResNet50, device)
except Exception as e:
    print(f"Failed to load CNN model: {e}")
    cnn_model = None

# Load ViT model
try:
    vit_model = load_model_with_state_dict("models/vit_model.pth", DualOutputViT, device)
except Exception as e:
    print(f"Failed to load ViT model: {e}")
    vit_model = None

# FIXED: Correct class labels for DR (5 classes) and DME (3 classes)
DR_CLASSES = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']  # 5 classes
DME_CLASSES = ['No DME', 'Mild DME', 'Severe DME']  # 3 classes

# Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def preprocess_image(filepath):
    img = Image.open(filepath).convert('RGB')
    img = preprocess(img)
    img = img.unsqueeze(0)  # Add batch dimension
    return img


def predict_model(model, img_tensor):
    with torch.no_grad():
        output = model(img_tensor)
        # If your model outputs two heads (DR, DME), split accordingly
        dr_logits, dme_logits = output
        dr_probs = torch.softmax(dr_logits, dim=1).cpu().numpy()[0]
        dme_probs = torch.softmax(dme_logits, dim=1).cpu().numpy()[0]
    return dr_probs, dme_probs


def get_prediction_results(model, img_tensor, model_name):
    """Get prediction results for a given model and image"""
    if model is None:
        print(f"❌ {model_name} model not loaded")
        return {
            'dr': {
                'label': 'No DR',
                'confidence': 0.0,
                'distribution': [1.0, 0.0, 0.0, 0.0, 0.0],  # 5 classes for DR
                'error': f'{model_name} model not loaded'
            },
            'dme': {
                'label': 'No DME',
                'confidence': 0.0,
                'distribution': [1.0, 0.0, 0.0],  # 3 classes for DME
                'error': f'{model_name} model not loaded'
            }
        }

    try:
        dr_probs, dme_probs = predict_model(model, img_tensor)

        # Debug: Print the raw predictions
        print(f"{model_name} DR predictions shape: {dr_probs.shape}, values: {dr_probs}")
        print(f"{model_name} DME predictions shape: {dme_probs.shape}, values: {dme_probs}")

        # Ensure we have the correct number of classes
        if len(dr_probs) != len(DR_CLASSES):
            print(
                f"⚠️  Warning: DR prediction length ({len(dr_probs)}) doesn't match DR_CLASSES length ({len(DR_CLASSES)})")
            # Pad or truncate as needed
            if len(dr_probs) < len(DR_CLASSES):
                dr_probs = np.pad(dr_probs, (0, len(DR_CLASSES) - len(dr_probs)), 'constant')
            else:
                dr_probs = dr_probs[:len(DR_CLASSES)]

        if len(dme_probs) != len(DME_CLASSES):
            print(
                f"⚠️  Warning: DME prediction length ({len(dme_probs)}) doesn't match DME_CLASSES length ({len(DME_CLASSES)})")
            # Pad or truncate as needed
            if len(dme_probs) < len(DME_CLASSES):
                dme_probs = np.pad(dme_probs, (0, len(DME_CLASSES) - len(dme_probs)), 'constant')
            else:
                dme_probs = dme_probs[:len(DME_CLASSES)]

        # Get predicted classes (highest probability)
        dr_pred_idx = np.argmax(dr_probs)
        dme_pred_idx = np.argmax(dme_probs)

        # Get labels and confidences
        dr_label = DR_CLASSES[dr_pred_idx]
        dme_label = DME_CLASSES[dme_pred_idx]
        dr_confidence = float(dr_probs[dr_pred_idx])
        dme_confidence = float(dme_probs[dme_pred_idx])

        # Ensure probabilities are properly normalized
        dr_probs = dr_probs / np.sum(dr_probs)
        dme_probs = dme_probs / np.sum(dme_probs)

        result = {
            'dr': {
                'label': dr_label,
                'confidence': dr_confidence,
                'distribution': dr_probs.tolist()
            },
            'dme': {
                'label': dme_label,
                'confidence': dme_confidence,
                'distribution': dme_probs.tolist()
            }
        }

        print(f"{model_name} Final result:")
        print(f"  DR: {dr_label} ({dr_confidence:.3f}), distribution: {[f'{p:.3f}' for p in dr_probs]}")
        print(f"  DME: {dme_label} ({dme_confidence:.3f}), distribution: {[f'{p:.3f}' for p in dme_probs]}")

        return result

    except Exception as e:
        print(f"Error in {model_name} prediction: {e}")
        import traceback
        traceback.print_exc()

        # Return mock data as fallback
        return {
            'dr': {
                'label': 'No DR',
                'confidence': 0.0,
                'distribution': [1.0, 0.0, 0.0, 0.0, 0.0],  # 5 classes for DR
                'error': str(e)
            },
            'dme': {
                'label': 'No DME',
                'confidence': 0.0,
                'distribution': [1.0, 0.0, 0.0],  # 3 classes for DME
                'error': str(e)
            }
        }


@app.route('/')
def index():
    print("Index route accessed")
    try:
        return render_template('index.html')
    except Exception as e:
        print(f"Error rendering template: {e}")
        return f"Error: {e}", 500


@app.route('/predict', methods=['POST'])
def predict():
    print("Predict route accessed")
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Save the uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        print(f"File saved to: {filepath}")

        # Preprocess the image
        img_tensor = preprocess_image(filepath)
        print(f"Image tensor shape: {img_tensor.shape}")

        # Get predictions from both models
        print("Getting CNN predictions...")
        cnn_results = get_prediction_results(cnn_model, img_tensor, "CNN")

        print("Getting ViT predictions...")
        vit_results = get_prediction_results(vit_model, img_tensor, "ViT")

        # Prepare results
        results = {
            'cnn': cnn_results,
            'vit': vit_results,
            'filename': file.filename
        }

        print("Final results structure:")
        print(f"CNN DR classes: {len(cnn_results['dr']['distribution'])}")
        print(f"CNN DME classes: {len(cnn_results['dme']['distribution'])}")
        print(f"ViT DR classes: {len(vit_results['dr']['distribution'])}")
        print(f"ViT DME classes: {len(vit_results['dme']['distribution'])}")

        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass

        return jsonify(results)

    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()

        # Return error response
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'filename': file.filename
        }), 500


if __name__ == '__main__':
    print("Starting Flask app...")
    print(f"Template folder: {app.template_folder}")
    print(f"Static folder: {app.static_folder}")
    print(f"DR Classes ({len(DR_CLASSES)}): {DR_CLASSES}")
    print(f"DME Classes ({len(DME_CLASSES)}): {DME_CLASSES}")

    # Check if templates directory exists
    if not os.path.exists('templates'):
        print("❌ Templates directory not found!")
        print("Creating templates directory...")
        os.makedirs('templates', exist_ok=True)

    # Check if static directory exists
    if not os.path.exists('static'):
        print("❌ Static directory not found!")
        print("Creating static directory...")
        os.makedirs('static', exist_ok=True)

    app.run(debug=True, host='0.0.0.0', port=5000)
