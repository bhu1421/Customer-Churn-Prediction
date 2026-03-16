import subprocess
import sys
import time

from config import FEATURES_PATH, MODEL_PATH


def run_command(command: str, name: str):
    print(f"Starting {name}...")
    try:
        return subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        print(f"Error starting {name}: {exc}")
        return None


def model_files_available() -> bool:
    return MODEL_PATH.exists() and FEATURES_PATH.exists()


if __name__ == "__main__":
    print("Customer Churn Prediction - Deployment")
    print("=" * 50)

    if not model_files_available():
        print("[ERROR] Missing model files in the model_files folder.")
        print("Please run: python train_and_save_model.py")
        sys.exit(1)

    print("[OK] Model files found")

    fastapi_process = run_command(
        "uvicorn app:app --reload --host 0.0.0.0 --port 8000",
        "FastAPI server",
    )
    if fastapi_process:
        print("[OK] FastAPI server started on http://localhost:8000")
        print("   - API docs: http://localhost:8000/docs")
        print("   - Health check: http://localhost:8000/health")

    time.sleep(3)

    streamlit_process = run_command(
        "streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0",
        "Streamlit app",
    )
    if streamlit_process:
        print("[OK] Streamlit app started on http://localhost:8501")

    print("\n[OK] Both servers are running.")
    print("Open your browser to: http://localhost:8501")
    print("API available at: http://localhost:8000")
    print("\nPress Ctrl+C to stop both servers")

    try:
        while True:
            if fastapi_process and fastapi_process.poll() is not None:
                print("[ERROR] FastAPI server stopped unexpectedly.")
                break
            if streamlit_process and streamlit_process.poll() is not None:
                print("[ERROR] Streamlit app stopped unexpectedly.")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping servers...")
    finally:
        if fastapi_process and fastapi_process.poll() is None:
            fastapi_process.terminate()
        if streamlit_process and streamlit_process.poll() is None:
            streamlit_process.terminate()
        print("[OK] Servers stopped")
