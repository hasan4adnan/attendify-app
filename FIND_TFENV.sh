#!/bin/bash
# Safe script to find tfenv without triggering iCloud

echo "🔍 Looking for tfenv..."
echo "======================"

# Check common locations (no recursive search)
LOCATIONS=(
    "$HOME/tfenv"
    "$HOME/.tfenv"
    "./tfenv"
    "$HOME/Documents/tfenv"
    "$HOME/Desktop/tfenv"
)

FOUND=false
for LOC in "${LOCATIONS[@]}"; do
    if [ -d "$LOC" ] && [ -f "$LOC/bin/activate" ]; then
        echo "✓ FOUND: $LOC"
        echo ""
        echo "To activate, run:"
        echo "  source $LOC/bin/activate"
        echo ""
        FOUND=true
    fi
done

if [ "$FOUND" = false ]; then
    echo "❌ tfenv not found in common locations."
    echo ""
    echo "Where did you create it? Common options:"
    echo "  - In your home directory: ~/tfenv"
    echo "  - In project directory: ./tfenv"
    echo "  - Somewhere else?"
    echo ""
    echo "If you remember creating it, check:"
    echo "  ls -la ~/tfenv"
    echo "  ls -la ./tfenv"
fi

