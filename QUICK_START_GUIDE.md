# Attendify - Quick Start Guide

Complete guide to run the Attendify face recognition system.

---

## 🚀 Quick Start (3 Steps)

### Step 1: Activate Environment
```bash
# Navigate to project directory
cd /Users/hasan/Documents/Github/attendify-app

# Deactivate conda base (if active)
conda deactivate

# Activate TensorFlow environment (it's in the project directory!)
source ./tfenv/bin/activate

# Verify you're in the right environment
# Your prompt should show: (tfenv)
# NOT: (tfenv) (base)
```

### Step 2: Run Face Registration (First Time Only)
```bash
python app/register_face.py
```

### Step 3: Run Face Recognition
```bash
python app/main.py
```

---

## 📋 Detailed Instructions

### Prerequisites Check

Before starting, verify your environment:

```bash
# 1. Check Python version (should be 3.10.x)
python --version

# 2. Check you're using tfenv Python (not conda)
which python
# Should show: /Users/hasan/tfenv/bin/python

# 3. Verify dependencies
python test_tensorflow.py
```

Expected output:
```
✓ TensorFlow 2.14.0 imported successfully
✓ GPU acceleration available (1 device(s))
✓ DeepFace imported successfully
✓ OpenCV X.X.X imported successfully
✓ NumPy X.X.X imported successfully
```

---

## 🎯 Running the Application

### Option A: Register a New Face

**When to use:** First time setup, or adding a new person

```bash
# 1. Activate environment
cd /Users/hasan/Documents/Github/attendify-app
conda deactivate
source ~/tfenv/bin/activate

# 2. Run registration
python app/register_face.py
```

**What happens:**
1. Webcam opens with an ellipse on screen
2. Follow 5 registration steps:
   - "Please look slightly up"
   - "Please look slightly down"
   - "Turn your head to the left"
   - "Turn your head to the right"
   - "Keep your face centered in the ellipse"
3. Face embedding is saved to `embeddings/` folder
4. Press 'q' to exit

**Output:** Creates `embeddings/student_embedding_YYYYMMDD_HHMMSS.npy`

---

### Option B: Run Face Recognition

**When to use:** Daily attendance checking

```bash
# 1. Activate environment
cd /Users/hasan/Documents/Github/attendify-app
conda deactivate
source ~/tfenv/bin/activate

# 2. Run recognition
python app/main.py
```

**What happens:**
1. Webcam opens with an ellipse on screen
2. System loads all embeddings from `embeddings/` folder
3. When a registered face is detected:
   - Ellipse border turns green
   - "Match found" message appears below ellipse
   - Console shows similarity score
4. When unregistered face is detected:
   - No message appears
   - Console shows similarity score (should be < 0.3)
5. Press 'q' to quit

---

## 🔧 Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'deepface'"

**Solution:**
```bash
source ~/tfenv/bin/activate
pip install deepface opencv-python
```

---

### Problem: NumPy/TensorFlow errors

**Solution:**
```bash
source ~/tfenv/bin/activate
pip install numpy==1.26.4
python test_tensorflow.py
```

---

### Problem: Using wrong Python (conda base instead of tfenv)

**Solution:**
```bash
# Deactivate conda completely
conda deactivate

# Activate tfenv
source ~/tfenv/bin/activate

# Verify
which python
# Should show: /Users/hasan/tfenv/bin/python
```

---

### Problem: Webcam doesn't open

**Solutions:**
1. Check camera permissions in System Settings → Privacy & Security → Camera
2. Close other apps using the camera (Zoom, FaceTime, etc.)
3. Try a different camera index (edit code: change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`)

---

### Problem: "Match found" appears for everyone

**Solution:**
- Old embeddings might be from the old system (128D instead of 512D)
- Delete old embeddings: `rm embeddings/*.npy`
- Re-register faces using the new system

---

## 📁 File Structure

```
attendify-app/
├── app/
│   ├── main.py              # Face recognition (attendance checking)
│   └── register_face.py    # Face registration
├── embeddings/              # Saved face embeddings (512D)
│   └── student_embedding_*.npy
├── test_tensorflow.py      # Test script
└── QUICK_START_GUIDE.md    # This file
```

---

## 🎮 Keyboard Controls

- **'q' key**: Quit the application (works in both registration and recognition)

---

## 📊 Understanding Similarity Scores

When running recognition, you'll see similarity scores in the console:

- **0.6 - 0.9**: Same person (registered) → Shows "Match found"
- **0.3 - 0.6**: Uncertain match → No message (threshold is 0.45)
- **< 0.3**: Different person → No message

---

## 🔄 Daily Workflow

**Morning Setup:**
```bash
cd /Users/hasan/Documents/Github/attendify-app
conda deactivate
source ~/tfenv/bin/activate
python app/main.py
```

**Adding New Student:**
```bash
cd /Users/hasan/Documents/Github/attendify-app
conda deactivate
source ~/tfenv/bin/activate
python app/register_face.py
```

---

## 💡 Pro Tips

1. **Always deactivate conda base first** - prevents environment conflicts
2. **Check your prompt** - should show only `(tfenv)`, not `(tfenv) (base)`
3. **Good lighting** - helps with face detection accuracy
4. **Face centered in ellipse** - better recognition results
5. **First run downloads model** - ArcFace model (~100MB) downloads automatically on first use

---

## 🆘 Quick Reference Commands

```bash
# Activate environment
conda deactivate && source ~/tfenv/bin/activate

# Test setup
python test_tensorflow.py

# Register face
python app/register_face.py

# Run recognition
python app/main.py

# Check embeddings
ls -lh embeddings/

# Delete all embeddings (start fresh)
rm embeddings/*.npy
```

---

## ✅ Pre-Flight Checklist

Before running the app, verify:

- [ ] Conda base is deactivated
- [ ] tfenv is activated
- [ ] Python version is 3.10.x
- [ ] `which python` shows tfenv path
- [ ] `python test_tensorflow.py` passes all checks
- [ ] Camera permissions are enabled
- [ ] No other apps are using the camera

---

**That's it! You're ready to use Attendify! 🎉**



