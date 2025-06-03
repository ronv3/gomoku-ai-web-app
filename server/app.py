"""Flask wrapper around gomoku_ai.py that
runs the potentially slow getTurn() call in a background
thread so the main Flask worker never blocks.

If getTurn() exceeds AI_TIMEOUT seconds we return HTTP 504.
You can bump AI_TIMEOUT or switch to ProcessPoolExecutor if the
computation is strongly CPU‑bound.
"""

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from flask import Flask, request, jsonify
from flask_cors import CORS
import gomoku_ai

# --- basic Flask setup -------------------------------------------------------
app = Flask(__name__)
CORS(app)                     # leave enabled for safety; proxy also works

# --- worker pool to keep the dev server responsive ---------------------------
EXECUTOR = ThreadPoolExecutor(max_workers=4)
AI_TIMEOUT = 25               # seconds

"""Compute the bot's reply move.

    Expected JSON body::
        {
          "board": [[0,0,...],[...],...],  # 15×15 ints
          "player": 2                      # 1 or 2
        }

    Returns::
        {"move": [row, col]} on success
        {"error": "msg"} with HTTP 4xx/5xx otherwise
    """
@app.route("/move", methods=["POST"])
def move():
    data = request.get_json(force=True)
    board  = data["board"]
    player = data["player"]

    future = EXECUTOR.submit(gomoku_ai.getTurn, board, player)
    try:
        row, col = future.result(timeout=AI_TIMEOUT)
    except FutureTimeoutError:
        return jsonify({"error": f"AI computation exceeded {AI_TIMEOUT}s"}), 504
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify({"move": [row, col]})

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True, threaded=True)
