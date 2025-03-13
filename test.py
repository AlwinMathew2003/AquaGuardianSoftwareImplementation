from flask import Flask, jsonify

app = Flask(__name__)

# Global variable to track detection status
run_detection = False

@app.route('/on', methods=['GET'])
def start_detection():
    global run_detection
    run_detection = True
    return jsonify({"message": "Detection started", "status": run_detection}), 200

@app.route('/off', methods=['GET'])
def stop_detection():
    global run_detection
    run_detection = False
    return jsonify({"message": "Detection stopped", "status": run_detection}), 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
