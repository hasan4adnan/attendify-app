#!/bin/bash
# SIMPLE start script - no searching, no iCloud prompts

echo "🚀 Attendify - Simple Start"
echo "============================"
echo ""
echo "This script assumes you've already activated tfenv."
echo "If not, run: source ./tfenv/bin/activate"
echo ""
read -p "Press Enter to continue (or Ctrl+C to exit)..."
echo ""

# Navigate to project
cd /Users/hasan/Documents/Github/attendify-app

# Check if TensorFlow is available
if ! python -c "import tensorflow" 2>/dev/null; then
    echo "❌ ERROR: TensorFlow not found!"
    echo ""
    echo "Please activate tfenv first:"
    echo "  source ./tfenv/bin/activate"
    echo ""
    exit 1
fi

echo "✓ TensorFlow found!"
echo ""

# Ask what to do
echo "What would you like to do?"
echo "1) Run Face Recognition (attendance checking)"
echo "2) Register New Face"
echo "3) Test Setup"
echo ""
read -p "Enter choice (1/2/3): " choice

case $choice in
    1)
        echo ""
        echo "🎯 Starting Face Recognition..."
        python app/main.py
        ;;
    2)
        echo ""
        echo "📝 Starting Face Registration..."
        python app/register_face.py
        ;;
    3)
        echo ""
        echo "🧪 Testing setup..."
        python test_tensorflow.py
        ;;
    *)
        echo "Invalid choice."
        exit 1
        ;;
esac

