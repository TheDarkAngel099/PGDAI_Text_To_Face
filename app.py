"""
Main Application Launcher
Launches both Backend (FastAPI) and Frontend (Streamlit) services
"""
import os
import sys
import subprocess
import time
import argparse
from pathlib import Path
import webbrowser
import signal

# Get project root
PROJECT_ROOT = Path(__file__).parent
BACKEND_DIR = PROJECT_ROOT / "backend"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

class AppLauncher:
    def __init__(self, backend_port=8000, frontend_port=8501, open_browser=True):
        self.backend_port = backend_port
        self.frontend_port = frontend_port
        self.open_browser = open_browser
        self.processes = []
    
    def check_dependencies(self):
        """Check if required packages are installed"""
        print("🔍 Checking dependencies...")
        
        # Check backend dependencies
        print("  Backend: ", end="")
        try:
            import fastapi
            import uvicorn
            import pydantic
            print("✅")
        except ImportError as e:
            print(f"❌ Missing: {e}")
            print(f"    Install with: pip install -r {BACKEND_DIR}/requirements.txt")
            return False
        
        # Check frontend dependencies
        print("  Frontend: ", end="")
        try:
            import streamlit
            import requests
            from PIL import Image
            print("✅")
        except ImportError as e:
            print(f"❌ Missing: {e}")
            print(f"    Install with: pip install -r {FRONTEND_DIR}/requirements.txt")
            return False
        
        return True
    
    def start_backend(self):
        """Start FastAPI backend server"""
        print(f"\n🚀 Starting Backend (FastAPI)...")
        print(f"   Port: {self.backend_port}")
        print(f"   Docs: http://localhost:{self.backend_port}/docs")
        
        backend_script = BACKEND_DIR / "main.py"
        
        if not backend_script.exists():
            print(f"❌ Backend script not found: {backend_script}")
            return False
        
        try:
            # Set environment to run backend using conda
            env = os.environ.copy()
            env['PYTHONPATH'] = str(BACKEND_DIR)
            
            # Try to use conda run with text_to_face environment
            process = subprocess.Popen(
                ["conda", "run", "-n", "text_to_face", "python", str(backend_script)],
                cwd=str(BACKEND_DIR),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            self.processes.append(("Backend", process))
            print("✅ Backend started")
            
            # Give backend time to start
            time.sleep(3)
            
            return True
        
        except Exception as e:
            print(f"❌ Failed to start backend: {e}")
            return False
    
    def start_frontend(self):
        """Start Streamlit frontend"""
        print(f"\n🎨 Starting Frontend (Streamlit)...")
        print(f"   App: Forensic Face Description System")
        print(f"   URL: http://localhost:{self.frontend_port}")
        
        frontend_script = FRONTEND_DIR / "forensic_app.py"
        
        if not frontend_script.exists():
            print(f"❌ Frontend script not found: {frontend_script}")
            return False
        
        try:
            # Use conda run to ensure correct environment
            process = subprocess.Popen(
                [
                    "conda", "run", "-n", "text_to_face", "python", "-m", "streamlit", "run",
                    str(frontend_script),
                    "--logger.level=info",
                    f"--server.port={self.frontend_port}",
                    "--server.address=localhost"
                ],
                cwd=str(FRONTEND_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            self.processes.append(("Frontend", process))
            print("✅ Frontend started")
            
            return True
        
        except Exception as e:
            print(f"❌ Failed to start frontend: {e}")
            return False
    
    def open_in_browser(self):
        """Open the application in default browser"""
        if self.open_browser:
            # Wait for frontend to be ready
            time.sleep(5)
            url = f"http://localhost:{self.frontend_port}"
            print(f"\n🌐 Opening {url} in default browser...")
            webbrowser.open(url)
    
    def run(self):
        """Start all services"""
        print("\n" + "="*60)
        print("🔍 FORENSIC FACE DESCRIPTION SYSTEM")
        print("="*60)
        
        # Check dependencies
        if not self.check_dependencies():
            print("\n❌ Missing dependencies. Please install them first:")
            print(f"   pip install -r backend/requirements.txt")
            print(f"   pip install -r frontend/requirements.txt")
            return False
        
        print("✅ All dependencies found")
        
        # Start services
        if not self.start_backend():
            print("\n❌ Failed to start backend")
            self.cleanup()
            return False
        
        if not self.start_frontend():
            print("\n❌ Failed to start frontend")
            self.cleanup()
            return False
        
        # Open in browser
        self.open_in_browser()
        
        # Print instructions
        print("\n" + "="*60)
        print("✅ ALL SERVICES RUNNING")
        print("="*60)
        print(f"\n📊 API Documentation:  http://localhost:{self.backend_port}/docs")
        print(f"🎨 Forensic App:       http://localhost:{self.frontend_port}")
        print(f"\n💡 Simple Generator:   http://localhost:{self.frontend_port}?page=streamlit_app")
        print("\nPress Ctrl+C to stop all services...")
        print("\n" + "="*60 + "\n")
        
        # Keep running
        try:
            while True:
                time.sleep(1)
                
                # Check if processes are still running
                for name, process in self.processes:
                    if process.poll() is not None:
                        print(f"\n⚠️  {name} process stopped (exit code: {process.returncode})")
        
        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down services...")
            self.cleanup()
            print("✅ All services stopped")
            return True
    
    def cleanup(self):
        """Stop all running processes"""
        for name, process in self.processes:
            try:
                print(f"  Stopping {name}...", end=" ")
                process.terminate()
                process.wait(timeout=5)
                print("✅")
            except subprocess.TimeoutExpired:
                process.kill()
                print("(killed)")
            except Exception as e:
                print(f"(error: {e})")


def main():
    parser = argparse.ArgumentParser(
        description="Launch Forensic Face Description System"
    )
    parser.add_argument(
        "--backend-port",
        type=int,
        default=8000,
        help="Backend API port (default: 8000)"
    )
    parser.add_argument(
        "--frontend-port",
        type=int,
        default=8501,
        help="Frontend Streamlit port (default: 8501)"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically"
    )
    parser.add_argument(
        "--backend-only",
        action="store_true",
        help="Start only backend API"
    )
    parser.add_argument(
        "--frontend-only",
        action="store_true",
        help="Start only frontend (requires backend running)"
    )
    
    args = parser.parse_args()
    
    launcher = AppLauncher(
        backend_port=args.backend_port,
        frontend_port=args.frontend_port,
        open_browser=not args.no_browser
    )
    
    # Handle specific modes
    if args.backend_only:
        print("\n" + "="*60)
        print("🔍 FORENSIC FACE DESCRIPTION SYSTEM - Backend Only")
        print("="*60)
        
        if not launcher.check_dependencies():
            print("\n❌ Missing dependencies")
            return
        
        if launcher.start_backend():
            print("\n" + "="*60)
            print("✅ BACKEND RUNNING")
            print("="*60)
            print(f"\n📊 API Documentation: http://localhost:{args.backend_port}/docs")
            print("\nPress Ctrl+C to stop...")
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n\n🛑 Shutting down...")
                launcher.cleanup()
                print("✅ Backend stopped")
    
    elif args.frontend_only:
        print("\n" + "="*60)
        print("🔍 FORENSIC FACE DESCRIPTION SYSTEM - Frontend Only")
        print("="*60)
        print(f"ℹ️  Make sure backend is running on http://localhost:{args.backend_port}")
        
        if not launcher.check_dependencies():
            print("\n❌ Missing dependencies")
            return
        
        if launcher.start_frontend():
            launcher.open_in_browser()
            print("\n" + "="*60)
            print("✅ FRONTEND RUNNING")
            print("="*60)
            print(f"\n🎨 Forensic App: http://localhost:{args.frontend_port}")
            print("\nPress Ctrl+C to stop...")
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n\n🛑 Shutting down...")
                launcher.cleanup()
                print("✅ Frontend stopped")
    
    else:
        # Run both
        launcher.run()


if __name__ == "__main__":
    main()
