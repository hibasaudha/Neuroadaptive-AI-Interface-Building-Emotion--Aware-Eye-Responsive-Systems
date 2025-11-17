import os
import time
from flask import Flask, render_template, jsonify, request, send_from_directory

# Try to import your existing logic. If import fails, the app will start but endpoints will return helpful errors.
try:
    from REALTIME import get_fused_emotion, get_realtime_face_emotion, get_realtime_eye_movement
except Exception as e:
    # Provide fallbacks so the frontend still runs for UI testing
    def _missing_fn(*args, **kwargs):
        raise RuntimeError("multimodal_inputs.get_fused_emotion or related functions are not importable: " + str(e))
    get_fused_emotion = _missing_fn
    get_realtime_face_emotion = lambda: "N/A"
    get_realtime_eye_movement = lambda: "N/A"

try:
    from output import get_cheer_up_response
except Exception as e:
    def get_cheer_up_response(user_query: str, fused_emotion: str) -> str:
        return f"(Stub) Would call Gemini with '{fused_emotion}' for: {user_query}"

# Configuration
CV_OUTPUT_FILE = "cv_output.txt"  # produced by your cv_server
TEMPLATE_DIR = "templates"

app = Flask(__name__, template_folder=TEMPLATE_DIR)

# Ensure templates folder exists and contains index.html
INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Multimodal Companion</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-50 min-h-screen flex items-center justify-center p-6">
  <div class="max-w-3xl w-full bg-white rounded-2xl shadow-lg p-6">
    <h1 class="text-2xl font-semibold mb-4">Multimodal Empathetic Companion</h1>

    <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
      <div class="col-span-1 p-4 bg-gray-100 rounded-lg">
        <h2 class="text-sm font-medium text-gray-600">Live Face Emotion</h2>
        <div id="face_emotion" class="mt-2 text-xl font-bold">N/A</div>
      </div>

      <div class="col-span-1 p-4 bg-gray-100 rounded-lg">
        <h2 class="text-sm font-medium text-gray-600">Eye State (EAR)</h2>
        <div id="eye_state" class="mt-2 text-xl font-bold">N/A</div>
      </div>

      <div class="col-span-1 p-4 bg-gray-100 rounded-lg">
        <h2 class="text-sm font-medium text-gray-600">Fused State (Preview)</h2>
        <div id="fused_preview" class="mt-2 text-xl font-bold">N/A</div>
      </div>
    </div>

    <form id="chatForm" class="mb-4">
      <label class="block text-sm font-medium text-gray-700">Send a message to the Companion</label>
      <textarea id="user_input" rows="3" class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:ring-indigo-500 focus:border-indigo-500" placeholder="How are you feeling?"></textarea>
      <div class="mt-3 flex items-center justify-between">
        <div class="text-sm text-gray-500">Type your message and press Send. The backend will run fusion and invoke the Gemini responder.</div>
        <div>
          <button type="submit" class="inline-flex items-center px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700">Send</button>
        </div>
      </div>
    </form>

    <div id="chat_area" class="bg-gray-50 border border-gray-100 rounded-lg p-4 h-56 overflow-auto">
      <div class="text-sm text-gray-500">Chat history...</div>
    </div>

    <div class="mt-4 text-right text-xs text-gray-400">Press Ctrl+R to refresh UI styles if tailwind CDN updates.</div>
  </div>

<script>
// Poll /api/state for live CV output
async function pollState() {
  try {
    const res = await fetch('/api/state');
    const data = await res.json();
    document.getElementById('face_emotion').innerText = data.face || 'N/A';
    document.getElementById('eye_state').innerText = data.eye || 'N/A';
    document.getElementById('fused_preview').innerText = data.fused_preview || 'N/A';
  } catch (e) {
    console.warn('State poll failed', e);
  }
}

setInterval(pollState, 1000);
pollState();

// Chat form
const chatForm = document.getElementById('chatForm');
const chatArea = document.getElementById('chat_area');
chatForm.addEventListener('submit', async (ev)=>{
  ev.preventDefault();
  const text = document.getElementById('user_input').value.trim();
  if (!text) return;

  // show user message
  const userNode = document.createElement('div');
  userNode.className = 'text-right mb-2';
  userNode.innerHTML = `<div class="inline-block bg-indigo-600 text-white px-3 py-2 rounded-lg">${text}</div>`;
  chatArea.appendChild(userNode);
  chatArea.scrollTop = chatArea.scrollHeight;

  // call backend
  const resp = await fetch('/api/chat', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text})
  });
  const j = await resp.json();

  const botNode = document.createElement('div');
  botNode.className = 'text-left mb-2';
  botNode.innerHTML = `<div class="inline-block bg-gray-200 text-gray-900 px-3 py-2 rounded-lg">${j.reply}</div><div class="text-xs text-gray-500">Fused: ${j.fused}</div>`;
  chatArea.appendChild(botNode);
  chatArea.scrollTop = chatArea.scrollHeight;

  document.getElementById('user_input').value = '';
});
</script>
</body>
</html>
"""


def ensure_templates():
    os.makedirs(TEMPLATE_DIR, exist_ok=True)
    index_path = os.path.join(TEMPLATE_DIR, 'index.html')
    if not os.path.exists(index_path):
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(INDEX_HTML)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/state')
def api_state():
    """Reads cv_output.txt and returns JSON {face, eye, fused_preview}.
    The fused_preview is a very lightweight mapping for UI convenience.
    """
    face = 'N/A'
    eye = 'N/A'
    fused_preview = 'N/A'

    # Read CV output file (format: EMOTION_LABEL|EAR_LABEL)
    try:
        if os.path.exists(CV_OUTPUT_FILE):
            with open(CV_OUTPUT_FILE, 'r') as f:
                data = f.read().strip()
                if '|' in data:
                    face, eye = data.split('|')
    except Exception as e:
        face = f'ERR'
        eye = 'ERR'

    # Lightweight fused preview (mirror of some simple rules)
    if face.lower() == 'sad' or eye.lower() == 'closed_eyes':
        fused_preview = 'Concern'
    elif face.lower() == 'happy':
        fused_preview = 'Positive'
    else:
        fused_preview = 'Neutral'

    return jsonify({'face': face, 'eye': eye, 'fused_preview': fused_preview})


@app.route('/api/chat', methods=['POST'])
def api_chat():
    """Receives JSON {text}, runs fusion + cheerup pipeline and returns reply.
    Uses your get_fused_emotion and get_cheer_up_response functions.
    """
    payload = request.get_json() or {}
    text = payload.get('text', '').strip()
    if not text:
        return jsonify({'error': 'empty message', 'reply': ''}), 400

    try:
        fused = get_fused_emotion(text)
    except Exception as e:
        fused = f'Error in fusion: {e}'

    try:
        reply = get_cheer_up_response(text, fused)
    except Exception as e:
        reply = f'Error generating reply: {e}'

    return jsonify({'fused': fused, 'reply': reply})


if __name__ == '__main__':
    ensure_templates()
    print('Starting Flask app. Templates ensured in ./templates/index.html')
    app.run(debug=True, host='0.0.0.0', port=5000)