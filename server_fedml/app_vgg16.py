import os
import sys
import gc
import json
import time
import glob
import shutil
import psutil
import logging
import traceback
import pickle
import io  
from datetime import datetime, timedelta
from functools import wraps
from logging.handlers import RotatingFileHandler
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import onnx
import onnxruntime as ort
from onnx_tf.backend import prepare
import tensorflowjs as tfjs
from tensorflow.keras.applications import VGG16 as KerasVGG16
from tensorflow.keras import layers as KL, models as KM

from flask import Flask, request, jsonify, g, make_response, send_file, send_from_directory
from flask_jwt_extended import JWTManager, jwt_required, get_jwt_identity, verify_jwt_in_request
from flask_cors import CORS

def build_keras_vgg16(num_classes=5):
    base = KerasVGG16(include_top=False, weights=None, input_shape=(224,224,3))
    x = base.output
    x = KL.Flatten(name='flatten')(x)
    x = KL.Dense(4096, activation='relu', name='fc1')(x)
    x = KL.Dropout(0.5)(x)
    x = KL.Dense(4096, activation='relu', name='fc2')(x)
    x = KL.Dropout(0.5)(x)
    out = KL.Dense(num_classes, activation='softmax', name='predictions')(x)
    return KM.Model(inputs=base.input, outputs=out)

def export_weights_to_npy(torch_state_dict, out_dir='vgg16_weights'):
    os.makedirs(out_dir, exist_ok=True)
    for key, tensor in torch_state_dict.items():
        if key.endswith(".weight") or key.endswith(".bias"):
            base = key.rsplit('.', 1)[0]
            kind = key.rsplit('.', 1)[1] 
            out_path = os.path.join(out_dir, f"{base}.{kind}.npy")
            np.save(out_path, tensor.numpy())

def load_npy_to_keras(keras_model, npy_dir='vgg16_weights'):
    conv_map = [
        ('features.0', 'block1_conv1'),
        ('features.2', 'block1_conv2'),
        ('features.5', 'block2_conv1'),
        ('features.7', 'block2_conv2'),
        ('features.10', 'block3_conv1'),
        ('features.12', 'block3_conv2'),
        ('features.14', 'block3_conv3'),
        ('features.17', 'block4_conv1'),
        ('features.19', 'block4_conv2'),
        ('features.21', 'block4_conv3'),
        ('features.24', 'block5_conv1'),
        ('features.26', 'block5_conv2'),
        ('features.28', 'block5_conv3'),
    ]

    dense_map = [
    ('classifier.0','fc1'), ('classifier.3','fc2'), ('classifier.6','predictions'),
    ]
    # load conv
    for pt_name, ks_name in conv_map:
        W = np.load(f"{npy_dir}/{pt_name}.weight.npy")
        b = np.load(f"{npy_dir}/{pt_name}.bias.npy")
        W = np.transpose(W, (2,3,1,0))
        layer = keras_model.get_layer(ks_name)
        layer.set_weights([W, b])
    # load dense
    for pt_name, ks_name in dense_map:
        W = np.load(f"{npy_dir}/{pt_name}.weight.npy")
        b = np.load(f"{npy_dir}/{pt_name}.bias.npy")
        # PyTorch linear is (out, in), Keras Dense expects (in, out)
        W = W.T  
        layer = keras_model.get_layer(ks_name)
        layer.set_weights([W, b])
        
    for layer in keras_model.layers:
        if layer.name in [name for _,name in dense_map]:
            w, b = layer.get_weights()
            print(layer.name, w.mean(), b.mean())
            
# ─── Logging Setup ─────────────────────────────────────────────────────────────
logger = logging.getLogger('fedml_server')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = RotatingFileHandler('fedml_server.log', maxBytes=10*1024*1024, backupCount=5)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# ─── Flask App Setup ───────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config['STATIC_DIR'] = os.path.join(BASE_DIR, 'static')


CORS(app, resources={r"/*": {
    "origins": ["http://localhost:5050","http://localhost:3000","http://localhost:5173", "https://baga-net.vercel.app", "https://baga-net-backend.vercel.app"],
    "methods": ["GET", "POST", "PUT", "OPTIONS"],
    "allow_headers": ["Content-Type", "X-API-Key", "X-Client-ID", "Authorization", 'ngrok-skip-browser-warning'],
    "supports_credentials": True
}})

@app.after_request
def add_cors_headers(response):
    origin = request.headers.get('Origin')
    allowed_origins = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:5173",
        "https://baga-net.vercel.app",
        "https://baga-net-backend.vercel.app",
        "https://381d-103-72-190-144.ngrok-free.app"
    ]

    if origin in allowed_origins:
        response.headers['Access-Control-Allow-Origin'] = origin

    requested = request.headers.get('Access-Control-Request-Headers')
    response.headers['Access-Control-Allow-Headers'] = (
        requested or
        'Accept, Authorization, Content-Type, X-API-Key, ngrok-skip-browser-warning'
    )

    requested_m = request.headers.get('Access-Control-Request-Method')
    response.headers['Access-Control-Allow-Methods'] = (
        requested_m or 'GET,OPTIONS'
    )

    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Max-Age'] = '3600'

    if request.method == 'OPTIONS':
        response.status_code = 200
        response.data = b''
        return response

    return response

# ─── Configuration ────────────────────────────────────────────────────────────
app.config['JWT_SECRET_KEY']       = os.environ.get('JWT_SECRET_KEY', 'default-dev-key')
app.config['MODEL_DIR']            = os.environ.get('MODEL_DIR','models')
app.config['CLIENTS_DB']           = os.environ.get('CLIENTS_DB','clients.json')
app.config['METRICS_DB']           = os.environ.get('METRICS_DB','metrics.json')
app.config['SKIP_TF_CONVERSIONS']  = os.environ.get('SKIP_TF_CONVERSIONS','true').lower() == 'true'
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 1 GB max upload size (already set)
app.config['SEND_FILE_MAX_LENGTH'] = 2 * 1024 * 1024 * 1024  # Allow larger file uploads
app.config['AGGREGATION_THRESHOLD'] = int(os.environ.get('AGGREGATION_THRESHOLD', 1))

API_KEY = os.environ.get('API_KEY','FeDMl2025')
os.makedirs(app.config['MODEL_DIR'], exist_ok=True)

# ─── JWT Setup ─────────────────────────────────────────────────────────────────
jwt = JWTManager(app)

# ─── Globals ─────────────────────────────────────────────────────────────────
global onnx_model_path, saved_model_dir, tfjs_model_dir
onnx_model_path = None
saved_model_dir = None
tfjs_model_dir = None

current_model_version = 0
model_path      = os.path.join(app.config['MODEL_DIR'], 'vgg16_state_dict.pth')
onnx_model_path = None
saved_model_dir = None
tfjs_model_dir  = None

# ─── ONNX Runtime Options ─────────────────────────────────────────────────────
ort_options = ort.SessionOptions()
ort_options.intra_op_num_threads = 2
ort_options.inter_op_num_threads = 1
ort_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
ort_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# ─── Request Timing & Logging ─────────────────────────────────────────────────
@app.before_request
def start_timer():
    g.start_time = time.time()
    logger.info(f"Incoming request: {request.method} {request.path} from {request.remote_addr}")

class VGG16(nn.Module):
    def __init__(self,num_classes=5):
        super().__init__()
        m = tv_models.vgg16(pretrained=False)
        m.classifier[6] = nn.Linear(4096,num_classes)
        self.features, self.avgpool, self.classifier = m.features, m.avgpool, m.classifier
    def forward(self,x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        return self.classifier(x)

@app.teardown_request
def log_exception(exc):
    if exc:
        logger.error(f"Exception during request: {exc}")
        logger.error(traceback.format_exc()) 

# ─── Utility Functions ────────────────────────────────────────────────────────
def log_memory_usage(tag):
    proc = psutil.Process(os.getpid())
    mb = proc.memory_info().rss / 1024**2
    logger.info(f"[{tag}] Memory usage: {mb:.1f} MB")

def read_db(path):
    try:
        with open(path,'r') as f: return json.load(f)
    except: return {}

def write_db(path,data):
    tmp = path + '.tmp'
    with open(tmp,'w') as f: json.dump(data,f)
    os.replace(tmp,path)

read_clients_db = lambda: read_db(app.config['CLIENTS_DB'])
write_clients_db = lambda d: write_db(app.config['CLIENTS_DB'],d)
read_metrics_db = lambda: read_db(app.config['METRICS_DB'])
write_metrics_db = lambda d: write_db(app.config['METRICS_DB'],d)

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.method == 'OPTIONS':
            return f(*args, **kwargs)
        
        
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key != "FeDMl2025":
            return jsonify({'message': 'Invalid or missing API key'}), 401
            
        return f(*args, **kwargs)
    
    return decorated_function

def require_auth_token(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        
        print(f"DEBUG - Received auth header: {auth_header}")
        
        if not auth_header:
            return jsonify({'message': 'Authorization header missing'}), 401
            
        if not auth_header.startswith('Bearer '):
            return jsonify({'message': 'Invalid token format, must start with "Bearer "'}), 401

        token = auth_header.split(' ')[1]
        
        try:
            import jwt as pyjwt
            try:
                payload = pyjwt.decode(token, options={"verify_signature": False})
                print(f"DEBUG - Token payload: {payload}")
            except Exception as e:
                print(f"DEBUG - Cannot decode token: {e}")
        except ImportError:
            print("DEBUG - PyJWT not installed, skipping token debug")

        try:
            verify_jwt_in_request()
            print("DEBUG - JWT validation successful")
            return f(*args, **kwargs)
        except Exception as e:
            print(f"DEBUG - JWT validation error: {str(e)}")
            return jsonify({'message': 'Invalid or expired token'}), 401
            
    return wrapper

def get_model_static_path(version):
    """Returns the versioned static path for model files"""
    return os.path.join(app.static_folder, 'tfjs', f'v{version}')

def weighted_fedavg(weights_list, sample_counts):
    # First convert all weights to consistent format
    converted_weights = []
    for weights in weights_list:
        converted = {}
        
        # Check if weights are in layer_X format (from TFJS)
        if any(k.startswith('layer_') for k in weights.keys()):
            # TFJS to PyTorch conversion mapping
            # This mapping needs to match your actual model architecture
            layer_mapping = {
                # Features layers
                'layer_0': 'features.0.weight', 'layer_1': 'features.0.bias',
                'layer_2': 'features.2.weight', 'layer_3': 'features.2.bias',
                'layer_4': 'features.5.weight', 'layer_5': 'features.5.bias',
                'layer_6': 'features.7.weight', 'layer_7': 'features.7.bias',
                'layer_8': 'features.10.weight', 'layer_9': 'features.10.bias',
                'layer_10': 'features.12.weight', 'layer_11': 'features.12.bias',
                'layer_12': 'features.14.weight', 'layer_13': 'features.14.bias',
                'layer_14': 'features.17.weight', 'layer_15': 'features.17.bias',
                'layer_16': 'features.19.weight', 'layer_17': 'features.19.bias',
                'layer_18': 'features.21.weight', 'layer_19': 'features.21.bias',
                'layer_20': 'features.24.weight', 'layer_21': 'features.24.bias',
                'layer_22': 'features.26.weight', 'layer_23': 'features.26.bias',
                'layer_24': 'features.28.weight', 'layer_25': 'features.28.bias',
                # Classifier layers
                'layer_26': 'classifier.0.weight', 'layer_27': 'classifier.0.bias',
                'layer_28': 'classifier.3.weight', 'layer_29': 'classifier.3.bias',
                'layer_30': 'classifier.6.weight', 'layer_31': 'classifier.6.bias'
            }
            
            for tfjs_name, torch_name in layer_mapping.items():
                if tfjs_name in weights:
                    converted[torch_name] = weights[tfjs_name]
        else:
            # Assume weights are already in PyTorch format
            converted = weights
            
        converted_weights.append(converted)
    
    # Now perform weighted averaging
    total_samples = sum(sample_counts)
    
    if total_samples == 0:
        logger.warning("Total sample count is zero - falling back to simple averaging")
        averaged = {}
        for key in converted_weights[0].keys():
            layer_weights = [w[key] for w in converted_weights]
            averaged[key] = sum(layer_weights) / len(layer_weights)
        return averaged
    
    # Weighted average
    averaged = {}
    for key in converted_weights[0].keys():
        weighted_sum = sum(w[key] * n for w, n in zip(converted_weights, sample_counts))
        averaged[key] = weighted_sum / total_samples
    
    return averaged
def update_metrics(metrics, version):
    """Update the metrics database with new metrics for the given version"""
    metrics_db = read_metrics_db()
    
    # Initialize if empty
    if 'history' not in metrics_db:
        metrics_db['history'] = []
    
    # Update current version
    metrics_db['current_version'] = version
    
    # Add new metrics entry with timestamp
    metrics_entry = {
        'version': version,
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics
    }
    
    metrics_db['history'].append(metrics_entry)
    
    # Limit history size to prevent endless growth
    if len(metrics_db['history']) > 20:
        metrics_db['history'] = metrics_db['history'][-20:]
    
    # Write updated metrics
    write_metrics_db(metrics_db)

# ─── Enhanced Conversion Functions ───────────────────────────────────────────
def export_to_onnx(model: torch.nn.Module, version: int) -> str:
    path = os.path.join(app.config['MODEL_DIR'], f'vgg16_v{version}.onnx')
    dummy = torch.zeros((1,3,224,224), dtype=torch.float32)
    try:
        torch.onnx.export(
            model, dummy, path,
            opset_version=11,
            input_names=['input'], output_names=['output'],
            dynamic_axes={'input':{0:'batch'}, 'output':{0:'batch'}}
        )
        onnx_model = onnx.load(path)
        onnx.checker.check_model(onnx_model)
        logger.info(f"ONNX saved: {path}")
        return path
    except Exception:
        if os.path.exists(path): os.remove(path)
        logger.error("Failed ONNX export", exc_info=True)
        raise



@app.before_first_request
def init_model():
    global current_model_version, onnx_model_path, saved_model_dir, tfjs_model_dir

    version = 1

    # PyTorch → ONNX
    pt = tv_models.vgg16(pretrained=False)
    pt.classifier[6] = nn.Linear(4096,5)
    state = torch.load(model_path, map_location='cpu')
    pt.load_state_dict(state); pt.eval()

    version = 1
    onnx_model_path = export_to_onnx(pt, version)


    # ─── export PyTorch weights to .npy, rebuild & load into Keras ────
    weights_npy_dir = os.path.join(app.config['MODEL_DIR'], f'vgg16_weights_v{version}')
    export_weights_to_npy(state, out_dir=weights_npy_dir)
    keras_model = build_keras_vgg16(num_classes=5)
    load_npy_to_keras(keras_model, npy_dir=weights_npy_dir)

    # export to TF.js Layers
    tfjs_layers_dir = os.path.join(app.config['MODEL_DIR'], f'tfjs_layers_v{version}')
    tfjs.converters.save_keras_model(keras_model, tfjs_layers_dir)
    tfjs_model_dir = tfjs_layers_dir
    static_dir = os.path.join(app.static_folder, 'tfjs', f'v{version}')
    os.makedirs(static_dir, exist_ok=True)
    for fname in os.listdir(tfjs_layers_dir):
        shutil.copy(os.path.join(tfjs_layers_dir, fname),
                    os.path.join(static_dir, fname))
    
    current_model_version = version
    logger.info("Model init complete: ONNX, SavedModel graph, AND TF.js layers all exported.")


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.route('/tfjs_model',methods=['GET'])
@require_api_key
def get_tfjs_file():
    global tfjs_model_dir


    # Check if TFJS model is available
    if not tfjs_model_dir:
        return jsonify({'message': 'TFJS model not available - conversion failed or skipped'}), 503
    
    fname=request.args.get('file','model.json')
    path=os.path.join(tfjs_model_dir,fname)
    if not os.path.exists(path): return jsonify({'message':'Not found'}),404
    return send_file(path,mimetype='application/octet-stream')

@app.route('/static/tfjs/v<version>/<path:filename>')
def tfjs_static_files(version, filename):
    # First check if directory exists
    static_dir = os.path.join(app.static_folder,'tfjs',f'v{version}')
    if not os.path.exists(static_dir):
        return jsonify({'message': f'TFJS model version {version} not available'}), 404
        
    return send_from_directory(static_dir, filename)


@app.route('/api/verify', methods=['POST'])
@require_api_key
def verify_api_key():
    """Verify if the API key is valid"""
    client_id = request.headers.get('X-Client-ID') or request.json.get('client_id', 'unknown')
    
    return jsonify({
        'message': 'API key verified successfully',
        'client_id': client_id,
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/api/token', methods=['POST', 'OPTIONS'])
@require_api_key
def get_token():
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, X-API-Key, X-Client-ID, Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        response.status_code = 200
        return response

    try:
        data = request.get_json()
        client_id = data.get('client_id', 'unknown')
        
        token = "example_token_12345"  
        
        return jsonify({
            'access_token': token,
            'token_type': 'bearer',
            'expires_in': 3600  # 1 hour
        })
    except Exception as e:
        return jsonify({'message': f'Error generating token: {str(e)}'}), 500


@app.route('/inference', methods=['POST'])
@require_api_key
def inference():
    """Memory-optimized inference using the current model"""
    # Check if we received the input data
    if 'image' not in request.files:
        return jsonify({'message': 'No image provided'}), 400
    
    try:
        log_memory_usage("before_inference")
        
        # Get the image from the request
        image_file = request.files['image']
        
        # Check if the current ONNX model exists
        if not os.path.exists(onnx_model_path):
            return jsonify({'message': 'Model not found'}), 404
        
        # Create an ONNX Runtime session with optimized settings
        session = ort.InferenceSession(
            onnx_model_path, 
            sess_options=ort_options,
            providers=['CPUExecutionProvider']  # Force CPU
        )
        
        # Process the image
        input_data = pickle.loads(image_file.read())  # Expecting serialized numpy array
        
        # Ensure input is float32 to save memory
        if input_data.dtype != np.float32:
            input_data = input_data.astype(np.float32)
        
        # Perform inference
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        result = session.run([output_name], {input_name: input_data})
        
        # Post-process results
        probabilities = np.exp(result[0]) / np.sum(np.exp(result[0]), axis=1, keepdims=True)
        predicted_class = np.argmax(probabilities, axis=1)[0]
        
        # Free resources
        del session, input_data, result
        gc.collect()
        
        log_memory_usage("after_inference")
        
        # Map class indices to names for better readability
        class_names = {
            0: "Edema",
            1: "Pneumothorax", 
            2: "COVID-19",
            3: "Normal",
            4: "Pneumonia"
        }
        
        class_name = class_names.get(int(predicted_class), f"Unknown ({predicted_class})")
        
        return jsonify({
            'predicted_class': int(predicted_class),
            'class_name': class_name,
            'probabilities': probabilities[0].tolist(),
            'model_version': current_model_version
        }), 200
        
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        logger.error(traceback.format_exc())  
        return jsonify({'message': f'Error during inference: {str(e)}'}), 500

@app.route('/model', methods=['GET', 'OPTIONS'])
@require_api_key
def get_model():
    # Handle OPTIONS requests
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, X-API-Key, X-Client-ID, Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        response.status_code = 200
        return response
    
    # Handle GET requests
    if not os.path.exists(onnx_model_path):
        return jsonify({'message': 'Model not found'}), 404
    
    try:
        return send_file(
            onnx_model_path,
            as_attachment=True,
            download_name=f'vgg16_v{current_model_version}.onnx',
            mimetype='application/octet-stream'
        )
    except Exception as e:
        return jsonify({'message': f'Error sending model file: {str(e)}'}), 500


# Alias for backward compatibility with React component
@app.route('/download_model', methods=['GET'])
@require_api_key
def download_model():
    """Alias for get_model to maintain compatibility with frontend"""
    return get_model()


@app.route('/submit_weights', methods=['POST'])
#@require_api_key
def submit_weights():

    print("Request Headers:", dict(request.headers))
    print("Form Data:", request.form)
    print("Files:", request.files)
    """Submit locally trained weights and metrics from a client"""
    client_id = request.headers.get('X-Client-ID') or 'unknown'
    
    if 'weights' not in request.files:
        return jsonify({'message': 'No weights file provided'}), 400
        print("No weights file provided")
    if 'metrics' not in request.form:
        return jsonify({'message': 'No metrics provided'}), 400
        print("No metrics provided")

    try:
        log_memory_usage("before_submit_weights")

        # Load and parse metrics
        metrics = json.loads(request.form['metrics'])

        # Load weights from JSON file upload
        weights_file = request.files['weights']
        
        if weights_file.content_type not in ['application/json', 'application/octet-stream']:
            return jsonify({'message': f'Invalid content type: {weights_file.content_type}. Expected JSON format.'}), 400
        
        # Parse the JSON data
        try:
            # Read the file content as string first
            weights_content = weights_file.read().decode('utf-8')
            # Parse the JSON
            weights_data = json.loads(weights_content)
        except UnicodeDecodeError:
            # Reset file pointer
            weights_file.seek(0)
            weights_data = json.load(weights_file)

        weights_torch = {}
        
        
        if isinstance(weights_data, dict):
            # Direct format (flattened weights)
            if not any(isinstance(v, dict) for v in weights_data.values()):
                for k, v in weights_data.items():
                    if isinstance(v, list) or isinstance(v, tuple):
                        weights_torch[k] = torch.tensor(v, dtype=torch.float32)
                    else:
                        # Single value case
                        weights_torch[k] = torch.tensor([v], dtype=torch.float32)
            
            # Nested format with 'layers'
            elif 'layers' in weights_data:
                for k, v in weights_data['layers'].items():
                    # Handle both cases: when v is a dict with 'data' or when v is the data array itself
                    if isinstance(v, dict) and 'data' in v:
                        if isinstance(v['data'], list) or isinstance(v['data'], tuple):
                            weights_torch[k] = torch.tensor(v['data'], dtype=torch.float32)
                        else:
                            # Single value case
                            weights_torch[k] = torch.tensor([v['data']], dtype=torch.float32)
                    else:
                        if isinstance(v, list) or isinstance(v, tuple):
                            weights_torch[k] = torch.tensor(v, dtype=torch.float32)
                        else:
                            # Single value case
                            weights_torch[k] = torch.tensor([v], dtype=torch.float32)
            
            # Nested format without 'layers'
            else:
                for k, v in weights_data.items():
                    if isinstance(v, dict):
                        # If nested structure, look for data field
                        data = v.get('data', v)
                        if isinstance(data, list) or isinstance(data, tuple):
                            weights_torch[k] = torch.tensor(data, dtype=torch.float32)
                        else:
                            # Single value case
                            weights_torch[k] = torch.tensor([data], dtype=torch.float32)
                    elif isinstance(v, list) or isinstance(v, tuple):
                        weights_torch[k] = torch.tensor(v, dtype=torch.float32)
                    else:
                        # Single value case
                        weights_torch[k] = torch.tensor([v], dtype=torch.float32)
        else:
            return jsonify({'message': 'Invalid weights format: Expected a dictionary'}), 400

        # Save weights in .pt format for later aggregation
        client_weights_path = os.path.join(app.config['MODEL_DIR'], f'client_{client_id}_weights.pt')
        torch.save(weights_torch, client_weights_path)

        # Update the metrics record for this client
        clients_db = read_clients_db()
        if client_id not in clients_db:
            clients_db[client_id] = {}

        clients_db[client_id]['last_contribution'] = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        write_clients_db(clients_db)

        log_memory_usage("after_submit_weights")

        return jsonify({'message': 'Weights and metrics submitted successfully'}), 200

    except json.JSONDecodeError as json_error:
        logger.error(f"JSON parsing error: {str(json_error)}")
        return jsonify({'message': f'Invalid JSON format: {str(json_error)}'}), 400
    except KeyError as key_error:
        logger.error(f"Missing key in weights data: {str(key_error)}")
        return jsonify({'message': f'Missing key in weights data: {str(key_error)}'}), 400
    except Exception as e:
        logger.error(f"Error processing submitted weights: {str(e)}")
        return jsonify({'message': f'Error processing submission: {str(e)}'}), 500

@app.route('/aggregate', methods=['POST'])
#@require_api_key
def aggregate_models():
    client_id = request.headers.get('X-Client-ID') or 'unknown'
    try:
        log_memory_usage("before_aggregation")
        model_dir = app.config['MODEL_DIR']
        weights_files = [f for f in os.listdir(model_dir) if f.startswith('client_') and f.endswith('_weights.pt')]
        if not weights_files:
            return jsonify({'message': 'No client weights available for aggregation'}), 400
        weights_list = []
        for wf in weights_files:
            try: weights_list.append(torch.load(os.path.join(model_dir,wf), map_location='cpu'))
            except: continue
        if not weights_list:
            return jsonify({'message': 'Failed to load any valid client weights'}),400
        # build sample counts
        metadata_dir = os.path.join(app.config['MODEL_DIR'],'client_metadata')
        sample_counts = []
        for wf in weights_files:
            meta_name = wf.replace('_weights', '_metadata').replace('.pt','.json')
            meta_path = os.path.join(metadata_dir, meta_name)
            if os.path.exists(meta_path):
                meta = json.load(open(meta_path))
                sample_counts.append(meta.get('num_samples',1))
            else:
                sample_counts.append(1)
        averaged = weighted_fedavg(weights_list, sample_counts)
        
        # Free memory
        del weights_list
        gc.collect()
        
        log_memory_usage("after_fedavg")
        
        # Increment version
        global current_model_version
        current_model_version += 1
        
        # Save the new model - without loading the full model to save memory
        new_model_path = os.path.join(app.config['MODEL_DIR'], f'vgg16_v{current_model_version}.pt')
        torch.save(averaged_weights, new_model_path)
        
        # Free memory
        del averaged_weights
        gc.collect()
        
        log_memory_usage("after_save_weights")
        
        # Load the model architecture and apply the averaged weights
        model = VGG16(num_classes=5)
        model.load_state_dict(torch.load(new_model_path, map_location='cpu'))
        model.eval()
        
        # Convert to ONNX
        global onnx_model_path
        onnx_model_path = export_to_onnx(model, current_model_version)
        
        # Free memory
        del model
        gc.collect()
        
        log_memory_usage("after_onnx_conversion")
        
        # Aggregate metrics from clients
        clients_db = read_clients_db()
        aggregated_metrics = {}
        for client, data in clients_db.items():
            if 'last_contribution' in data and 'metrics' in data['last_contribution']:
                client_metrics = data['last_contribution']['metrics']
                for metric_name, value in client_metrics.items():
                    if metric_name not in aggregated_metrics:
                        aggregated_metrics[metric_name] = []
                    aggregated_metrics[metric_name].append(value)
        
        # Average the metrics
        final_metrics = {metric: sum(values) / len(values) for metric, values in aggregated_metrics.items() if values}
        
        # Update metrics history
        update_metrics(final_metrics, current_model_version)
        
        # Clean up client weights files
        for weights_file in weights_files:
            try:
                os.remove(os.path.join(model_dir, weights_file))
            except Exception as e:
                logger.warning(f"Could not delete client weights file {weights_file}: {str(e)}")
        
        log_memory_usage("after_aggregation")
        
        return jsonify({'message':'Models aggregated successfully','new_version':current_model_version}),200
    except Exception as e:
        logger.error(f"Error during aggregation: {e}")
        return jsonify({'message':f'Error during aggregation: {e}'}),500
    

@app.route('/metrics', methods=['GET'])
@require_api_key
def get_metrics():
    """Get current model metrics"""
    
    metrics_data = read_metrics_db()
    
    # Get the latest metrics
    if metrics_data['history']:
        latest_metrics = metrics_data['history'][-1]
    else:
        latest_metrics = {'version': current_model_version, 'metrics': {}}
    
    return jsonify({
        'current_version': current_model_version,
        'latest_metrics': latest_metrics,
        'history': metrics_data['history'][-5:] if request.args.get('include_history') == 'true' else None
    }), 200

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with memory usage information"""
    # Check if model exists
    model_exists = os.path.exists(onnx_model_path)
    
    # Get memory usage using psutil
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    # Get CPU usage
    cpu_percent = psutil.cpu_percent(interval=0.1)
    
    # Get system memory info
    system_memory = psutil.virtual_memory()
    
    # Get disk usage
    disk_usage = psutil.disk_usage('/')
    
    return jsonify({
        'status': 'healthy' if model_exists else 'degraded',
        'model_version': current_model_version,
        'model_available': model_exists,
        'server_time': datetime.now().isoformat(),
        'memory_usage': {
            'process_rss_mb': memory_info.rss / 1024 / 1024,
            'system_used_percent': system_memory.percent,
            'system_available_mb': system_memory.available / 1024 / 1024
        },
        'cpu_usage': {
            'percent': cpu_percent,
            'cores': psutil.cpu_count()
        },
        'disk_usage': {
            'percent': disk_usage.percent,
            'free_gb': disk_usage.free / 1024 / 1024 / 1024
        }
    }), 200 if model_exists else 503

@app.route('/tfjs/model', methods=['GET'])
@require_api_key
@jwt_required(optional=True)
def get_tfjs_model_info():
    """Get current TFJS model version and metadata with proper HTTPS URLs"""
    try:
        global current_model_version
        version = current_model_version
        
        # Build path to the versioned model directory
        model_dir = get_model_static_path(version)
        
        # Check if model exists
        model_json_path = os.path.join(model_dir, 'model.json')
        if not os.path.exists(model_json_path):
            return jsonify({'message': 'TFJS model not found'}), 404
            
        # Get metrics for the current model
        metrics_data = read_metrics_db()
        current_metrics = {}
        
        if 'history' in metrics_data and metrics_data['history']:
            for entry in reversed(metrics_data['history']):
                if entry.get('version') == version:
                    current_metrics = entry.get('metrics', {})
                    break
        
        # List all model files
        model_files = []
        if os.path.exists(model_dir):
            model_files = [f for f in os.listdir(model_dir) 
                           if f.endswith('.json') or f.endswith('.bin')]
        
        # Use request.scheme and host for fully qualified URLs
        base_url = f"{request.scheme}://{request.host}"
        
        # Build model info response with versioned URLs
        model_info = {
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'format': 'tfjs',
            'metrics': current_metrics,
            'files': model_files,
            'url_base': f"{base_url}/static/tfjs/v{version}/",
            'model_url': f"{base_url}/static/tfjs/v{version}/model.json",
            'expires': (datetime.now() + timedelta(days=7)).isoformat()
        }
        
        return jsonify(model_info), 200
        
    except Exception as e:
        logger.error(f"Error retrieving TFJS model info: {str(e)}")
        return jsonify({
            'message': f'Error retrieving model info',
            'error': str(e)
        }), 500
    
@app.route('/tfjs/training_model', methods=['GET','OPTIONS'])
@require_api_key
@jwt_required(optional=True)
def send_tfjs_model():
    if request.method == 'OPTIONS':
        resp = make_response('', 200)
        resp.headers.update({
            "Access-Control-Allow-Origin": "http://localhost:5173",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "Accept, Authorization, Content-Type, X-API-Key, ngrok-skip-browser-warning"
        })
        return resp

    version = current_model_version
    model_url = f"/static/tfjs/v{version}/model.json"
    static_path = os.path.join(app.config['STATIC_DIR'], 'tfjs', f'v{version}', 'model.json')
    if not os.path.exists(static_path):
        return jsonify({"message": "TFJS model not available"}), 503

    client_id = get_jwt_identity() or request.headers.get('X-Client-ID','anonymous')
    logger.info(f"Client {client_id} requested TFJS Layers model v{version}")
    clients_db = read_clients_db()
    clients_db.setdefault(client_id, {})['last_model_request'] = {
        'timestamp': datetime.now().isoformat(),
        'version': version,
        'ip_address': request.remote_addr
    }
    write_clients_db(clients_db)

    return jsonify({
        "version": version,
        "model_url": model_url,
        "class_names": ["Edema","Pneumothorax","COVID-19","Normal","Pneumonia"],
        "training_params": {
            "batch_size": 2,
            "learning_rate": 1e-4,
            "epochs": 5,
            "min_local_samples": 10
        }
    }), 200



@app.route('/static/tfjs/v<int:version>/<path:filename>')
def serve_tfjs_file(version, filename):
    dirpath = os.path.join(app.config['MODEL_DIR'], f'tfjs_layers_v{version}')
    return send_from_directory(dirpath, filename, max_age=0)
    
@app.route('/tfjs/submit_weights', methods=['POST'])
@jwt_required(optional=True)
def tfjs_submit_weights():
    print("Request Headers:", dict(request.headers))
    print("Form Data:", request.form)
    print("Files:", request.files)
    
    # Get client ID
    client_id = request.headers.get('X-Client-ID') or 'anonymous'
    
    try:
        log_memory_usage("before_tfjs_submit_weights")
        
        # Handle multipart form data
        if 'weights' not in request.files:
            return jsonify({'message': 'No weights file provided'}), 400
        
        if 'metadata' not in request.form:
            return jsonify({'message': 'No metadata provided'}), 400
        
        # Parse metadata
        metadata = json.loads(request.form['metadata'])
        metrics = metadata.get('metrics', {})
        client_round_num = metadata.get('round', 0)  # Client's reported round (may be incorrect)
        num_samples = metadata.get('num_samples', 0)
        timestamp = metadata.get('timestamp', datetime.now().isoformat())
        system_metrics = metadata.get('system_metrics', {})
        
        # Get the actual current round for this client
        clients_db = read_clients_db()
        if client_id not in clients_db:
            clients_db[client_id] = {'training_round': 0}
        elif 'training_round' not in clients_db[client_id]:
            clients_db[client_id]['training_round'] = 0
            
        # Increment the client's round number
        current_round = clients_db[client_id]['training_round']
        clients_db[client_id]['training_round'] += 1
        write_clients_db(clients_db)
        
        # Rest of your processing code remains the same until saving files:
        # Process weights file
        weights_file = request.files['weights']
        weights_content = weights_file.read()
        
        # Convert binary weights to numpy array
        weights_array = np.frombuffer(weights_content, dtype=np.float32).copy()
        
        # Get model architecture info
        model_file = request.files.get('model')
        if model_file:
            model_info = json.loads(model_file.read().decode('utf-8'))
            weight_specs = model_info.get('weightSpecs', [])
        else:
            weight_specs = []
        
        # Reconstruct weights into proper shape
        weights_torch = {}
        offset = 0
        for spec in weight_specs:
            name = spec['name']
            shape = spec['shape']
            size = np.prod(shape)

            # Extract the weights for this layer
            layer_weights = weights_array[offset:offset+size]

            # Validate length
            if layer_weights.size != size:
                raise ValueError(
                    f"Layer '{name}': expected {size} elements, but got {layer_weights.size}. "
                    f"Check that the weights match the model architecture."
                )

            # Reshape and convert to tensor
            try:
                reshaped = layer_weights.reshape(shape)
            except Exception as reshape_error:
                raise ValueError(
                    f"Reshape failed for layer '{name}' with shape {shape}: {reshape_error}"
                )

            weights_torch[name] = torch.from_numpy(reshaped)
            offset += size

        if offset != len(weights_array):
            logger.warning(
                f"Unused weights detected: expected {offset} elements, got {len(weights_array)}."
            )

        # Save weights - use the server-tracked round number
        client_weights_dir = os.path.join(app.config['MODEL_DIR'], 'client_weights')
        os.makedirs(client_weights_dir, exist_ok=True)
        weights_filename = f'client_{client_id}_weights_r{current_round}.pt'
        weights_path = os.path.join(client_weights_dir, weights_filename)
        torch.save(weights_torch, weights_path)
        
        # Save metadata - use server-tracked round number
        client_metadata_dir = os.path.join(app.config['MODEL_DIR'], 'client_metadata')
        os.makedirs(client_metadata_dir, exist_ok=True)
        metadata_filename = f'client_{client_id}_metadata_r{current_round}.json'
        with open(os.path.join(client_metadata_dir, metadata_filename), 'w') as f:
            json.dump({
                'client_id': client_id,
                'round': current_round,  # Use server-tracked round
                'client_reported_round': client_round_num,  # Keep client's reported round for reference
                'num_samples': num_samples,
                'metrics': metrics,
                'timestamp': timestamp,
                'system_metrics': system_metrics
            }, f)
        
        # Update clients DB with latest contribution
        clients_db[client_id]['last_contribution'] = {
            'timestamp': timestamp,
            'round': current_round,
            'num_samples': num_samples,
            'metrics': metrics,
            'system_metrics': system_metrics or {}
        }
        write_clients_db(clients_db)
        
        log_memory_usage("after_tfjs_submit_weights")
        
        print(f"TFJS-SUBMIT_WEIGHTS: Successfully received weights from client '{client_id}' for round {current_round}")

        # Check if we've reached the aggregation threshold for this round
        client_weights = glob.glob(os.path.join(client_weights_dir, f'client_*_weights_r{current_round}.pt'))

        if current_round >= 3:
            logger.info(f"Aggregation threshold reached ({len(client_weights)} submissions). Triggering aggregation for round {current_round}.")
            
            print(f"Aggregation threshold reached ({len(client_weights)} submissions). Triggering aggregation for round {current_round}.")
            # Trigger aggregation in a background thread
            from threading import Thread
            aggregation_thread = Thread(
                target=perform_aggregation,
                kwargs={'round_num': current_round}
            )
            aggregation_thread.start()
            
            return jsonify({
                'message': 'Weights submitted successfully. Aggregation started.',
                'aggregation_triggered': True,
                'client_id': client_id,
                'round': current_round,
                'timestamp': datetime.now().isoformat()
            }), 200
        else:
            print(f"Weights submitted successfully. Need {remaining} more submissions for aggregation.")
            remaining = app.config['AGGREGATION_THRESHOLD'] - len(client_weights)
            return jsonify({
                'message': f'Weights submitted successfully. Need {remaining} more submissions for aggregation.',
                'aggregation_triggered': False,
                'client_id': client_id,
                'round': current_round,
                'timestamp': datetime.now().isoformat()
            }), 200
        
    except Exception as e:
        logger.error(f"Error processing submission: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'message': f'Error processing submission: {str(e)}'}), 500

def perform_aggregation(round_num):
    try:
        log_memory_usage("before_perform_aggregation")
        client_id = "auto_aggregator"
        
        cw_dir = os.path.join(app.config['MODEL_DIR'], 'client_weights')
        cm_dir = os.path.join(app.config['MODEL_DIR'], 'client_metadata')
        weights_files = glob.glob(os.path.join(cw_dir, f'client_*_weights_r{round_num}.pt'))
        
        if not weights_files:
            logger.warning(f"No client weights found for round {round_num}")
            return False

        weights_list, sample_counts = [], []
        for wf in weights_files:
            try:
                cid = os.path.basename(wf).split('_')[1]
                md_path = os.path.join(cm_dir, f'client_{cid}_metadata_r{round_num}.json')
                
                weights = torch.load(wf, map_location='cpu')
                if not isinstance(weights, dict):
                    raise ValueError("Weights file does not contain a dictionary")
                weights_list.append(weights)
                
                cnt = 1
                if os.path.exists(md_path):
                    with open(md_path) as f:
                        md = json.load(f)
                        cnt = max(1, int(md.get('num_samples', 1)))
                sample_counts.append(cnt)
            except Exception as e:
                logger.error(f"Error processing {wf}: {e}")
                continue
                
        if not weights_list:
            logger.error("No valid client weights loaded")
            return False

        agg = weighted_fedavg(weights_list, sample_counts)
        
        agg = fix_conv_weights(agg)

        required_keys = {
            'features.0.weight', 'features.0.bias',
            'features.2.weight', 'features.2.bias',
            'features.5.weight', 'features.5.bias',
            'features.7.weight', 'features.7.bias',
            'features.10.weight', 'features.10.bias',
            'features.12.weight', 'features.12.bias',
            'features.14.weight', 'features.14.bias',
            'features.17.weight', 'features.17.bias',
            'features.19.weight', 'features.19.bias',
            'features.21.weight', 'features.21.bias',
            'features.24.weight', 'features.24.bias',
            'features.26.weight', 'features.26.bias',
            'features.28.weight', 'features.28.bias',
            'classifier.0.weight', 'classifier.0.bias',
            'classifier.3.weight', 'classifier.3.bias',
            'classifier.6.weight', 'classifier.6.bias'
        }
        
        missing_keys = required_keys - set(agg.keys())
        if missing_keys:
            logger.error(f"Missing required keys in aggregated weights: {missing_keys}")
            return False

        global current_model_version
        current_model_version += 1
        new_pt = os.path.join(app.config['MODEL_DIR'], f'vgg16_v{current_model_version}.pt')
        torch.save(agg, new_pt)
        
        model = VGG16(num_classes=5)
        model.load_state_dict(agg)
        model.eval()

        log_memory_usage("after_save_weights")

        m = VGG16(num_classes=5)
        m.load_state_dict(torch.load(new_pt, map_location='cpu'))
        m.eval()
        
        global onnx_model_path
        onnx_model_path = export_to_onnx(m, current_model_version)
        del m
        gc.collect()

        log_memory_usage("after_onnx_conversion")
        new_ver = current_model_version + 1
        weights_npy_dir = os.path.join(app.config['MODEL_DIR'], f'vgg16_weights_v{new_ver}')
        export_weights_to_npy(torch.load(new_pt, map_location='cpu'), out_dir=weights_npy_dir)
        keras_model = build_keras_vgg16(num_classes=5)
        load_npy_to_keras(keras_model, npy_dir=weights_npy_dir)
        tfjs_layers_dir = os.path.join(app.config['MODEL_DIR'], f'tfjs_layers_v{new_ver}')
        tfjs.converters.save_keras_model(keras_model, tfjs_layers_dir)

        static_dir = get_model_static_path(new_ver)
        os.makedirs(static_dir, exist_ok=True)
        for f in glob.glob(os.path.join(tfjs_layers_dir, '*')):
            shutil.copy(f, static_dir)

        global tfjs_model_dir
        tfjs_model_dir = tfjs_layers_dir

        aggregated_metrics = {}
        clients_db = read_clients_db()
        for cid, data in clients_db.items():
            if 'last_contribution' in data and data['last_contribution'].get('round') == round_num:
                client_metrics = data['last_contribution'].get('metrics', {})
                for metric_name, value in client_metrics.items():
                    if metric_name not in aggregated_metrics:
                        aggregated_metrics[metric_name] = []
                    aggregated_metrics[metric_name].append(value)

        final_metrics = {metric: sum(values) / len(values) for metric, values in aggregated_metrics.items() if values}
        update_metrics(final_metrics, current_model_version)

        for wf in weights_files:
            try:
                os.remove(wf)
                cid = os.path.basename(wf).split('_')[1]
                md_file = os.path.join(cm_dir, f'client_{cid}_metadata_r{round_num}.json')
                if os.path.exists(md_file):
                    os.remove(md_file)
            except Exception as e:
                logger.warning(f"Could not clean up files for {wf}: {e}")

        log_memory_usage("after_aggregation_cleanup")
        
        logger.info(f"Aggregation completed for round {round_num}. New model version: {new_ver}")
        return True

    except Exception as e:
        logger.error(f"Error during automatic aggregation for round {round_num}: {e}")
        logger.error(traceback.format_exc())
        return False

def fix_conv_weights(weights):
    fixed_weights = {}
    for k, v in weights.items():
        if 'weight' in k and v.ndim == 4:
            fixed_weights[k] = v.permute(3, 2, 0, 1).contiguous()
        elif 'weight' in k and v.ndim == 2:
            fixed_weights[k] = v.t().contiguous()
        else:
            fixed_weights[k] = v
    return fixed_weights


@app.route('/tfjs/aggregate', methods=['POST'])
@require_api_key
def tfjs_aggregate_models():
    """Manual aggregation endpoint - now just triggers the perform_aggregation function"""
    client_id = request.headers.get('X-Client-ID') or 'admin'
    try:
        if not request.is_json:
            return jsonify({'message': 'Request must be JSON'}), 400
            
        data = request.get_json()
        round_num = data.get('round', 0)  # Default to round 0 if not specified
        
        # Start aggregation in background
        from threading import Thread
        aggregation_thread = Thread(
            target=perform_aggregation,
            kwargs={'round_num': round_num}
        )
        aggregation_thread.start()
        
        return jsonify({
            'message': 'Aggregation started successfully',
            'round': round_num,
            'triggered_by': client_id,
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error triggering aggregation: {e}")
        return jsonify({
            'message': 'Error triggering aggregation',
            'error': str(e)
        }), 500
        
@app.route('/tfjs/metrics', methods=['GET'])
@require_api_key
@jwt_required(optional=True)
def get_tfjs_model_metrics():
    """
    Get metrics for the current or specified TFJS model version
    Returns accuracy, F1 score, precision, recall, and other performance metrics
    """
    try:
        # Get client ID if available
        client_id = request.headers.get('X-Client-ID') or 'anonymous'
        
        # Check if a specific version is requested
        version = request.args.get('version', None)
        if not version:
            # Use the current model version if none specified
            global current_model_version
            version = current_model_version
        else:
            # Convert version to integer
            try:
                version = int(version)
            except ValueError:
                return jsonify({'message': 'Invalid version format'}), 400
        
        # Read metrics from database
        metrics_data = read_metrics_db()
        version_metrics = {}
        
        # Find metrics for the requested version
        if 'history' in metrics_data and metrics_data['history']:
            for entry in metrics_data['history']:
                if entry.get('version') == version:
                    version_metrics = entry.get('metrics', {})
                    break
        
        # Create a standardized metrics response (with default zeros if no metrics found)
        standard_metrics = {
            'accuracy': version_metrics.get('accuracy', 0),
            'f1_score': version_metrics.get('f1_score', 0),
            'precision': version_metrics.get('precision', 0),
            'recall': version_metrics.get('recall', 0),
            'loss': version_metrics.get('loss', 0)
        }
        
        # Add any additional metrics that may be available
        additional_metrics = {}
        for key, value in version_metrics.items():
            if key not in standard_metrics.keys():
                additional_metrics[key] = value
        
        # Log the metrics request
        logger.info(f"Client {client_id} requested metrics for model v{version}")
        
        # Build the response
        response = {
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'metrics': standard_metrics,
            'additional_metrics': additional_metrics,
            'available': bool(version_metrics)  
        }
        
        # Add class-specific metrics if available
        if 'class_metrics' in version_metrics:
            response['class_metrics'] = version_metrics['class_metrics']
        
        # Create a mapping for class names
        class_names = {
            0: "Edema",
            1: "Pneumothorax", 
            2: "COVID-19",
            3: "Normal",
            4: "Pneumonia"
        }
        response['class_names'] = class_names
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error retrieving TFJS model metrics: {str(e)}")
        logger.error(traceback.format_exc())
        
        default_response = {
            'version': version if 'version' in locals() else 1,
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'accuracy': 0,
                'f1_score': 0,
                'precision': 0,
                'recall': 0,
                'loss': 0
            },
            'additional_metrics': {},
            'available': False,
            'class_names': {
                0: "Edema",
                1: "Pneumothorax", 
                2: "COVID-19",
                3: "Normal",
                4: "Pneumonia"
            }
        }
        
        return jsonify(default_response), 200
    
def update_detailed_metrics(metrics, version, class_metrics=None):
    """
    Update metrics database with detailed metrics including per-class metrics
    
    Args:
        metrics (dict): Dictionary of global metrics (accuracy, f1, etc.)
        version (int): Model version
        class_metrics (dict, optional): Per-class metrics
    """
    metrics_data = read_metrics_db()
    
    if 'history' not in metrics_data:
        metrics_data['history'] = []
    
    existing_entry = None
    for entry in metrics_data['history']:
        if entry.get('version') == version:
            existing_entry = entry
            break
    
    timestamp = datetime.now().isoformat()
    
    if existing_entry:
        existing_entry['metrics'].update(metrics)
        existing_entry['updated_at'] = timestamp
        if class_metrics:
            existing_entry['metrics']['class_metrics'] = class_metrics
    else:
        # Create a new entry
        new_entry = {
            'version': version,
            'metrics': metrics,
            'created_at': timestamp,
            'updated_at': timestamp
        }
        
        if class_metrics:
            new_entry['metrics']['class_metrics'] = class_metrics
            
        metrics_data['history'].append(new_entry)
    
    metrics_data['current'] = {
        'version': version,
        'metrics': metrics.copy()
    }
    
    if class_metrics:
        metrics_data['current']['metrics']['class_metrics'] = class_metrics
    
    write_metrics_db(metrics_data)
    
    return True


def update_metrics(metrics, version, class_metrics=None):
    """
    Update metrics in the database for the specified model version
    
    Args:
        metrics (dict): Dictionary of metrics
        version (int): Model version
        class_metrics (dict, optional): Per-class metrics dictionary
    """
    return update_detailed_metrics(metrics, version, class_metrics)


@app.errorhandler(Exception)
def handle_error(e):
    logger.error(f"Error: {str(e)}")
    return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS']='2'
    app.run(host='0.0.0.0', port=5050, threaded=True)
