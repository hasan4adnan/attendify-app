#!/bin/bash
# Quick start script for Attendify
# Run this script to start face recognition

echo "🚀 Starting Attendify Face Recognition System..."
echo "================================================"

# Navigate to project directory
cd /Users/hasan/Documents/Github/attendify-app

# Try to deactivate conda (gracefully handle if it fails)
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "📦 Deactivating conda base environment..."
    conda deactivate 2>/dev/null || {
        echo "   ⚠️  Conda deactivation skipped (not critical)"
        # Unset conda variables manually
        unset CONDA_DEFAULT_ENV
        unset CONDA_PREFIX
        unset CONDA_PROMPT_MODIFIER
    }
fi

# Try common tfenv locations (NO find command to avoid iCloud prompts)
# tfenv is in the project directory!
TFENV_PATHS=(
    "./tfenv/bin/activate"
    "$HOME/tfenv/bin/activate"
    "$HOME/.tfenv/bin/activate"
)

TFENV_FOUND=false
for TFENV_PATH in "${TFENV_PATHS[@]}"; do
    if [ -f "$TFENV_PATH" ]; then
        echo "🔧 Activating TensorFlow environment..."
        echo "   Found at: $TFENV_PATH"
        source "$TFENV_PATH"
        TFENV_FOUND=true
        break
    fi
done

# If tfenv not found, ask user directly (NO automatic searching)
if [ "$TFENV_FOUND" = false ]; then
    echo "⚠️  tfenv not found in common locations."
    echo ""
    echo "Please activate your tfenv manually first, then run this script again."
    echo ""
    echo "Common commands:"
    echo "  source ~/tfenv/bin/activate"
    echo "  source ~/.tfenv/bin/activate"
    echo ""
    echo "Or enter the full path to tfenv/bin/activate:"
    read -p "Path (or press Enter to exit): " CUSTOM_TFENV
    if [ -n "$CUSTOM_TFENV" ] && [ -f "$CUSTOM_TFENV" ]; then
        source "$CUSTOM_TFENV"
        TFENV_FOUND=true
    else
        echo "Exiting. Please activate tfenv manually first."
        exit 1
    fi
fi

# Verify environment
PYTHON_PATH=$(which python)
echo "✓ Using Python: $PYTHON_PATH"

echo "✓ Environment ready!"
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
        echo "   - Press 'q' to quit"
        echo ""
        python app/main.py
        ;;
    2)
        echo ""
        echo "📝 Starting Face Registration..."
        echo "   - Follow on-screen instructions"
        echo "   - Press 'q' to quit"
        echo ""
        python app/register_face.py
        ;;
    3)
        echo ""
        echo "🧪 Testing setup..."
        python test_tensorflow.py
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "✅ Done!"



