# Attendify - Smart Attendance System

Face recognition-based attendance system for automated classroom attendance tracking.

## 🚀 Quick Start

### Option 1: Use the Start Script (Easiest)
```bash
cd /Users/hasan/Documents/Github/attendify-app
./START_HERE.sh
```

### Option 2: Manual Start
```bash
# 1. Activate environment
cd /Users/hasan/Documents/Github/attendify-app
conda deactivate
source ~/tfenv/bin/activate

# 2. Run recognition
python app/main.py
```

## 📖 Full Documentation

See [QUICK_START_GUIDE.md](QUICK_START_GUIDE.md) for complete instructions.

## 🎯 Features

- **Face Detection**: Real-time face detection within an ellipse region
- **Face Registration**: Multi-angle face registration with 5-step process
- **Face Recognition**: Real-time face recognition with similarity matching
- **GPU Acceleration**: Uses Apple Silicon GPU via TensorFlow Metal
- **512D Embeddings**: Uses ArcFace (InsightFace) for discriminative face embeddings

## 📁 Project Structure

```
attendify-app/
├── app/
│   ├── main.py              # Face recognition module
│   └── register_face.py    # Face registration module
├── embeddings/              # Saved face embeddings
├── START_HERE.sh           # Quick start script
├── QUICK_START_GUIDE.md    # Complete guide
└── README.md               # This file
```

## 🔧 Requirements

- Python 3.10 (in tfenv virtual environment)
- TensorFlow 2.14+
- DeepFace
- OpenCV
- NumPy 1.26.4

## 📝 Usage

### Register a Face
```bash
source ~/tfenv/bin/activate
python app/register_face.py
```

### Run Recognition
```bash
source ~/tfenv/bin/activate
python app/main.py
```

## 🆘 Troubleshooting

See [QUICK_START_GUIDE.md](QUICK_START_GUIDE.md) for troubleshooting steps.

## 📄 License

[Your License Here]
