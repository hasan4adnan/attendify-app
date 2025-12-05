# Manual Start Guide (If Script Doesn't Work)

## Step 1: Find Your tfenv Location

Run this command to find where tfenv is:

```bash
find ~ -name "tfenv" -type d 2>/dev/null
```

Or check common locations:
```bash
ls -la ~/tfenv
ls -la ~/.tfenv
ls -la ./tfenv
```

## Step 2: Activate tfenv Manually

Once you find the path, activate it:

```bash
# Example (replace with your actual path):
source /path/to/tfenv/bin/activate

# Common locations:
# source ~/tfenv/bin/activate
# source ~/.tfenv/bin/activate
# source ./tfenv/bin/activate
```

## Step 3: Verify Environment

```bash
# Check Python version (should be 3.10.x)
python --version

# Check Python path (should show tfenv)
which python

# Test TensorFlow
python -c "import tensorflow; print('TensorFlow works!')"
```

## Step 4: Run the Application

```bash
# Navigate to project
cd /Users/hasan/Documents/Github/attendify-app

# Run recognition
python app/main.py

# OR run registration
python app/register_face.py
```

## Quick Commands (Copy & Paste)

```bash
# Find tfenv
find ~ -name "tfenv" -type d 2>/dev/null

# Activate (use path from above)
source /FOUND/PATH/tfenv/bin/activate

# Run app
cd /Users/hasan/Documents/Github/attendify-app
python app/main.py
```

