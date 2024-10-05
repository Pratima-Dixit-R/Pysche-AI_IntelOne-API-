import os
import sys
import subprocess
import torch

# Function to install dependencies using pip
def install_dependencies():
    print("Installing required dependencies...")
    packages = ['torch', 'transformers', 'openvino', 'modin[ray]', 'intel_extension_for_pytorch', 'pandas', 'scikit-learn-intelex']
    
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    print("All dependencies are installed.")

# Check if the required libraries are installed; if not, install them
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from openvino.runtime import Core
    import modin.pandas as pd  # Modin replaces pandas for faster dataframe operations
    import intel_extension_for_pytorch as ipex  # Intel Extension for PyTorch for optimization
    from sklearnex import patch_sklearn  # AI Analytics Toolkit: Optimized Scikit-learn
    patch_sklearn()  # Apply optimizations to sklearn
except ImportError:
    install_dependencies()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from openvino.runtime import Core
    import modin.pandas as pd
    import intel_extension_for_pytorch as ipex
    from sklearnex import patch_sklearn  # AI Analytics Toolkit: Optimized Scikit-learn
    patch_sklearn()

# Load LLaMA 3.0 model for conversational generation
def load_llama_model(device):
    model_name = "meta-llama/Llama-3b-chat-hf"  # LLaMA 3.0 from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Load model on the appropriate device (GPU if available, otherwise CPU)
    model = model.to(device)
    
    # Optimize model with Intel Extension for PyTorch (IPEX)
    model = ipex.optimize(model)

    return model, tokenizer

# Load Emotion Sensitivity Model from Hugging Face
def load_emotion_model():
    emotion_model_name = "bhadresh-savani/distilbert-base-uncased-emotion"  # NLP model for emotion detection
    emotion_model = pipeline("sentiment-analysis", model=emotion_model_name)
    
    return emotion_model

# Generate therapeutic conversation based on LLaMA 3.0 model
def generate_response(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)  # Move input tensors to the right device
    outputs = model.generate(inputs.input_ids, max_length=150)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

# Analyze the emotional state of user input
def analyze_emotion(emotion_model, user_input):
    emotions = emotion_model(user_input)
    dominant_emotion = max(emotions, key=lambda x: x['score'])
    
    return dominant_emotion['label'], dominant_emotion['score']

# Optional: Optimize the model using OpenVINO for Intel hardware (if required)
def optimize_model_with_openvino(model):
    core = Core()
    compiled_model = core.compile_model(model=model, device_name="CPU")  # Optimization for Intel CPUs
    
    return compiled_model

# Main loop for interacting with Psyche AI
def psyche_ai():
    print("Welcome to Psyche AI: Your Mental Wellness Buddy App\n")
    
    # Detect if GPU is available, otherwise fallback to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")

    # Load the LLaMA 3.0 model for conversation and Emotion model for emotion sensitivity
    llama_model, llama_tokenizer = load_llama_model(device)
    emotion_model = load_emotion_model()
    
    # Optional: Optimize LLaMA model with OpenVINO for Intel hardware (uncomment if needed)
    # llama_model = optimize_model_with_openvino(llama_model)

    print("Hello! I'm Psyche AI. How are you feeling today?\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye! Take care of yourself.")
            break
        
        # Step 1: Analyze user's emotional state using NLP model
        emotion_label, emotion_score = analyze_emotion(emotion_model, user_input)
        print(f"Detected emotion: {emotion_label} with confidence {emotion_score:.2f}")
        
        # Step 2: Generate a therapeutic response using LLaMA 3.0
        response = generate_response(llama_model, llama_tokenizer, user_input, device)
        
        print(f"Psyche AI: {response}\n")

# Start the Psyche AI mental wellness buddy app
if _name_ == "_main_":
    psyche_ai()
