#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if Python 3 is installed
if ! command_exists python3; then
    echo "Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Create a virtual environment
python3 -m venv env

# Activate the virtual environment
source env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install the required packages
pip install -r requirements.txt

# Provide instructions to the user
echo "Virtual environment setup complete. Use 'source env/bin/activate' to activate the virtual environment."
echo "To deactivate the virtual environment, use the command 'deactivate'."
