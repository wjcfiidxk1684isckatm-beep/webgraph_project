# signal_logger.py
from flask import Flask, request, jsonify
from datetime import datetime
import json
import os

app = Flask(__name__)
LOG_PATH = os.path.join("uploads", "webhook.log")
os.makedirs("uploads", exist_ok=True)

@app.route('/webhook', methods=['POST'])
def webhook():
    if not request.is_json:
        return jsonify({"error": "Invalid Content-Type"}), 400

    data = request.get_json()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = {
        "timestamp": timestamp,
        "data": data
    }

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    action = data.get("action")
    if action == "buy":
        print("✅ 買い注文を検出しました")
    elif action == "sell":
        print("🔻 売り注文を検出しました")
    else:
        print("⚠️ 未知のアクションです")

    print(f"[{timestamp}] 受信データ: {data}")
    return jsonify({"status": "received"}), 200

if __name__ == '__main__':
    app.run(port=5000)
