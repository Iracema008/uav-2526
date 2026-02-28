#!/bin/bash

# updated
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_ROOT/depthai_env"
MAIN_SCRIPT="$PROJECT_ROOT/auto_uav.py"

# Function to create a virtual environment if it doesn't exist
create_venv_if_not_exists() {

    # Check if the virtual environment directory exists
    if [ ! -d "$VENV_DIR" ]; then
        echo "Virtual environment not found. Creating a new one..."
        
        # updated: create isolated venv to avoid PEP 668
        python3 -m venv --copies "$VENV_DIR"
        
        if [ $? -ne 0 ]; then
            echo "Error: Failed to create virtual environment."
            return 1
        fi
        
        echo "Virtual environment created successfully."
    else
        echo "Virtual environment already exists."
    fi
}

# Function to install dependencies from requirements.txt
install_requirements() {

    # Activate the virtual environment
    source "$VENV_DIR/bin/activate"
    
    # Check if requirements.txt exists
    if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
        echo "Installing dependencies from requirements.txt..."
        
        # updated: upgrade pip and install packages safely
        pip install --upgrade pip
        pip install -r "$PROJECT_ROOT/requirements.txt"
        
        if [ $? -ne 0 ]; then
            echo "Error: Failed to install dependencies."
            deactivate
            return 1
        fi

        echo "Dependencies installed successfully."
    else
        echo "requirements.txt file not found. Skipping dependency installation."
    fi

    # Deactivate virtual environment
    deactivate
}

# Function to initialize the virtual environment
init_venv() {

    # Create the virtual environment if it doesn't exist
    create_venv_if_not_exists
    
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment. Exiting."
        return 1
    fi

    # Install the requirements in the virtual environment
    install_requirements
    
    if [ $? -ne 0 ]; then
        echo "Failed to install dependencies. Exiting."
        return 1
    fi
}

# Function to run the main.py script inside the virtual environment
run() {

    export PYTHONDONTWRITEBYTECODE=1

    init_venv

    # Check if the virtual environment exists
    if [ ! -d "$VENV_DIR" ]; then
        echo "Error: Virtual environment not found."
        return 1
    fi

    # Check if main.py exists
    if [ ! -f "$MAIN_SCRIPT" ]; then
        echo "Error: main.py not found."
        return 1
    fi

    # Activate the virtual environment
    source "$VENV_DIR/bin/activate"

    # updated: set PYTHONPATH for project folders
    export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/pixhawk_testing:$PROJECT_ROOT/vision:$PYTHONPATH"
    
    # Run the main.py script
    echo "Running main.py inside virtual environment..."
    python -u "$MAIN_SCRIPT"
    
    # Capture the exit code of the script
    local EXIT_CODE=$?

    # Deactivate the virtual environment
    deactivate

    # Return the exit code of the script
    return $EXIT_CODE
}

# Usage message to show how to use the functions
usage() {
    echo "Usage: source helper.sh"
    echo "Then call the following functions:"
    echo " - init_venv           : Create/activate venv and install requirements."
    echo " - run                 : Run main.py inside the virtual environment."
}

# If the script is executed directly, show usage instructions
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    usage
fi