## 🧠 Neuroadaptive Emotion Aware System

### Real-time Emotional and Cognitive State Modulation of LLMs

-----

### **Overview**

The **Neuroadaptive Emotion Aware System** is an innovative project that enhances conversational AI by integrating real-time computer vision with advanced Large Language Models (LLMs) via the Gemini API. The system monitors a user's **emotional state** (via Facial Emotion Recognition, FER) and **cognitive state** (via Eye Aspect Ratio, EAR) to dynamically adjust the LLM's tone, empathy, and conversational strategy. It is specifically designed to act as a supportive assistant, helping **high screen-time professionals** manage stress and prevent fatigue/burnout.

### **Core Functionality**

1.  **Dual-Modality Sensing:** Uses a standard webcam for parallel, low-latency extraction of two crucial features:
      * **FER (Affective State):** A lightweight CNN classifies discrete emotions.
      * **EAR (Cognitive State):** Geometric analysis of facial landmarks detects fatigue/attentiveness.
2.  **Adaptive Logic Layer:** Fuses the FER and EAR outputs into a single **Unified State Object** (e.g., "Frustrated + Fatigued").
3.  **Dynamic Prompt Generation:** Translates the Unified State Object into a specific **System Prompt** (e.g., "Respond with a calm, de-escalating tone...").
4.  **LLM Modulation:** Sends the dynamic prompt and user text to the **Gemini API** to generate a contextually empathetic response.

### **System Architecture**

The system employs a multi-component architecture ensuring real-time performance and data privacy:

  * **Frontend (UI):** Built using a web framework (e.g., HTML/JS/React) for user input, displaying the chat, and managing the webcam stream. Includes a transparent **UI Feedback** panel for the inferred state.
  * **Backend (Logic & Vision):** Built with Python (Flask/FastAPI) to host the Vision Pipeline. Crucially, all computationally intensive FER and EAR processing is performed here, adhering to **edge processing** principles.
  * **Vision Pipeline:** Python libraries (OpenCV, dlib, TensorFlow/PyTorch) handle frame processing and feature extraction.
  * **LLM Service:** Secured communication with the **Gemini API** handles all generative output.

### **Getting Started**

#### **Prerequisites**

  * Python 3.8+
  * A Google AI API Key (for the Gemini API)
  * A working webcam

#### **Installation**

1.  **Clone the Repository:**

    ```bash
    git clone [Your Repository URL Here]
    cd neuroadaptive-emotion-aware-system
    ```

2.  **Set up Virtual Environment and Install Dependencies:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use: .\venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Configure API Key:**
    Set your Gemini API key as an environment variable (crucial for security):

    ```bash
    export GEMINI_API_KEY="YOUR_API_KEY_HERE"
    ```

4.  **Run the Application:**
    Start the backend server (e.g., Flask/FastAPI):

    ```bash
    python main.py
    ```

    (Note: Specific command depends on your final backend framework.)

5.  **Access the Frontend:**
    Open your web browser and navigate to `http://localhost:5000` (or the port specified by your server). Grant camera permissions to start sensing.

### **Ethical Considerations & Privacy**

  * **No Recording:** Raw video frames are processed on the local machine and immediately discarded. **No video data is ever stored or transmitted.**
  * **Edge Processing:** All sensitive visual analysis occurs on the client's side, ensuring privacy and scalability.
  * **User Control:** A prominent **Pause/Disable** button is included in the UI, allowing users to instantly stop the visual sensing at any time.

### **Future Work**

  * Integration of **Voice and Prosody Analysis** for robust multimodal sensing.
  * Expansion to proactive **Cognitive Augmentation** (e.g., workload adjustment).
  * Further **model pruning** for enhanced scalability and reduced latency.
