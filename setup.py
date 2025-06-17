# setup.py
import subprocess
import sys
import os
import shutil
import json
import datetime

# --- NEW: GLOBAL INSTALLATION MARKERS ---
INSTALL_MARKER_FILE = ".installed_marker"
LICENSE_FILE = "license.json"
PROGRAM_INSTALL_ID = "program_installation_date"
# ---------------------------------------

def run_command(command, message):
    print(f"\n--- {message} ---")
    try:
        # Use subprocess.check_call for better error handling
        subprocess.check_call(command, shell=True)
        print(f"--- {message} complete! ---")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command '{e.cmd}' failed with exit code {e.returncode}")
        print(f"Output: {e.output.decode(errors='ignore')}") # Try to decode output
        print("Please check the output for details.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during: {message}")
        print(f"Error: {e}")
        sys.exit(1)

def load_license_data():
    if os.path.exists(LICENSE_FILE):
        with open(LICENSE_FILE, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: {LICENSE_FILE} is corrupted. Returning empty data.")
                return {}
    return {}

def save_license_data(data):
    with open(LICENSE_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def main():
    print("🚀 Starting AI Document Assistant Setup 🚀")
    print("This script will set up your Python environment and dependencies.")
    print("Please ensure you have Python 3.9+ installed and added to your PATH.")

    # --- NEW: Check for existing installation marker ---
    if os.path.exists(INSTALL_MARKER_FILE):
        print(f"\nWARNING: Installation marker '{INSTALL_MARKER_FILE}' found.")
        print("This program appears to be already installed. Re-running setup.py will NOT reset the trial period.")
        print("If you wish to reinstall (e.g., for development), please manually delete the '.installed_marker' file and 'license.json' (if present) and the 'simple_db' folder.")
        print("Exiting setup. Please run 'run_app.bat' to start the application.")
        sys.exit(0) # Exit gracefully as it's already installed
    # ----------------------------------------------------

    # 1. Create and activate virtual environment
    venv_dir = "venv"
    if os.path.exists(venv_dir):
        print(f"Existing virtual environment '{venv_dir}' found. Skipping creation.")
    else:
        run_command([sys.executable, "-m", "venv", venv_dir], "Creating virtual environment")

    # Determine activation script based on OS
    if sys.platform == "win32":
        activate_script = os.path.join(venv_dir, "Scripts", "activate.bat")
        python_executable = os.path.join(venv_dir, "Scripts", "python.exe")
        pip_executable = os.path.join(venv_dir, "Scripts", "pip.exe")
    else: # Fallback for Linux/macOS, though instructions are for Windows
        activate_script = os.path.join(venv_dir, "bin", "activate")
        python_executable = os.path.join(venv_dir, "bin", "python")
        pip_executable = os.path.join(venv_dir, "bin", "pip")

    # 2. Install Python packages using the virtual environment's pip
    packages = [
        "streamlit",
        "pymupdf",
        "sentence-transformers",
        "chromadb",
        "ollama",
        "Pillow",
        "requests"
    ]
    run_command(f'"{pip_executable}" install {" ".join(packages)}', "Installing Python packages")

    # 3. Create a simple batch file to run the Streamlit app
    run_app_bat_content = f"""@echo off
rem This script activates the virtual environment and runs the Streamlit app.

call "{activate_script}"
if %errorlevel% neq 0 (
    echo Failed to activate virtual environment. Make sure Python and venv are set up correctly.
    pause
    exit /b %errorlevel%
)

echo Starting Streamlit app...
"{python_executable}" -m streamlit run main.py
if %errorlevel% neq 0 (
    echo Streamlit app failed to start. Check for errors above.
    pause
    exit /b %errorlevel%
)
pause
"""
    with open("run_app.bat", "w") as f:
        f.write(run_app_bat_content)
    print("\n'run_app.bat' created. You can use this to start the application easily.")

    # --- NEW: Create installation marker and record global install date ---
    try:
        # Create the installation marker file
        with open(INSTALL_MARKER_FILE, 'w') as f:
            f.write(f"Installed on: {datetime.date.today().strftime('%Y-%m-%d')}")
        print(f"Created installation marker file: {INSTALL_MARKER_FILE}")

        # Record global install date in license.json
        license_data = load_license_data()
        if PROGRAM_INSTALL_ID not in license_data:
            license_data[PROGRAM_INSTALL_ID] = {
                "install_date": datetime.date.today().strftime("%Y-%m-%d"),
                "status": "initial_install"
            }
            save_license_data(license_data)
            print(f"Recorded program installation date in {LICENSE_FILE}.")
        else:
            print(f"Program installation date already recorded in {LICENSE_FILE}.")

    except Exception as e:
        print(f"ERROR: Could not create installation marker or record date: {e}")
        print("This may affect trial period functionality. Please check file permissions.")
        sys.exit(1)
    # ---------------------------------------------------------------------

    print("\n--- Python Environment Setup Complete! ---")
    print("Next, you need to set up Ollama and download the LLM model.")
    print("Please run the 'install_ollama_and_model.bat' file.")
    print("Once both steps are done, use 'run_app.bat' to start the application.")

if __name__ == "__main__":
    main()
