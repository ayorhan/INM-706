# setup.sh
#!/bin/bash

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install the requirements
pip install -r requirements.txt

echo "Setup complete. To activate the virtual environment, run 'source venv/bin/activate'."
