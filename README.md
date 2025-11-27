# Activity-Recognition-System-for-Surveillance-CNN-LSTM-
A deep-learning–based activity recognition system built using CNN + LSTM architecture for detecting suspicious human activities in surveillance environments. This model identifies actions such as fighting, loitering, running, and trespassing from video sequences in real time.

# Project Overview
This project combines Convolutional Neural Networks (CNN) for spatial feature extraction and Long Short-Term Memory (LSTM) networks for temporal sequence modeling.
The system is optimized for real-time CCTV surveillance, offering fast inference, accurate detection, and automated alerting.

# Key Features
1) Detects suspicious activities with 82% accuracy and 78% F1-score.
2) Trained on 12K+ video frames and 500+ activity clips.
3) Real-time inference at 15–18 FPS with <180 ms latency.
4) Uses OpenCV for frame extraction, preprocessing, and video analytics.
5) Sliding-window sequence modeling for temporal pattern recognition.
6) Automated alert generation reduces manual monitoring by 40%.

# Model Architecture
1) CNN (feature extraction from frames)
2) LSTM (temporal sequence learning)
3) Sequence length: 20–30 frames
4) Loss: Categorical Cross-Entropy
5) Optimizer: Adam

# Tech Stack
1) Python 
2) TensorFlow / Keras 
3) OpenCV 
4) NumPy, Pandas, Matplotlib 
5) Scikit-learn


# Project Structure
├── dataset/
│   ├── fight/
│   ├── loitering/
│   ├── trespassing/
│   └── running/
├── preprocessing/
│   ├── extract_frames.py
│   ├── normalize_frames.py
│   └── create_sequences.py
├── model/
│   ├── cnn_lstm_model.ipynb
│   ├── train.py
│   └── evaluate.py
├── real_time/
│   └── realtime_inference.py
├── requirements.txt
└── README.md


# Contact
# For queries or collaboration:
Name - Neelesh Singh
📩 Email: neeleshchoursiya0786@gmail.com
🔗 LinkedIn: https://www.linkedin.com/in/neelesh-singh-data-ml
