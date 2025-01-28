# server.py
# usage: python3 server.py
# starts server on port 5000
from flask import Flask, request, jsonify, send_from_directory
import subprocess
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


@app.route('/')
def serve_index():
    return send_from_directory('..', 'deprecated code/index-deprecated.html')


@app.route('/<path:filename>')
def serve_file(filename):
    return send_from_directory('..', filename)


@app.route('/save-csv', methods=['POST'])
def save_csv():
    try:
        data = request.json
        filename = data['filename']
        content = data['content']

        # Save the CSV file in the current directory
        with open(filename, 'w') as f:
            f.write(content)

        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/generate-visualization', methods=['POST'])
def generate_visualization():
    try:
        # Run the visualization script in the current directory
        subprocess.run(['python', 'polar-area-chart-v2.py'], check=True)
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    app.run(port=5000)
