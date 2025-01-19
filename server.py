from flask import Flask, request, jsonify, send_from_directory
import pathlib
import os
import subprocess
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('./<path:filename>')
def serve_assessment(filename):
    return send_from_directory('./', filename)

@app.route('/save-csv', methods=['POST'])
def save_csv():
    try:
        data = request.json
        save_location = data['location']
        filename = data['filename']
        content = data['content']
        
        # Ensure the directory exists
        os.makedirs(save_location, exist_ok=True)
        
        # Create full file path
        file_path = os.path.join(save_location, filename)
        
        # Save the CSV file
        with open(file_path, 'w') as f:
            f.write(content)
        
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/generate-visualization', methods=['POST'])
def generate_visualization():
    try:
        data = request.json
        save_location = data['location']
        
        # Change to the save location directory
        os.chdir(save_location)
        
        # Run the visualization script
        subprocess.run(['python', 'polar-area-chart-v2.py'], check=True)
        
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)