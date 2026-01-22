"""
System Verification Script
Run this to verify your Forensic Face Description System is properly set up
"""

import os
import sys
from pathlib import Path

def check_structure():
    """Verify project structure"""
    print("\n" + "="*60)
    print("🔍 FORENSIC FACE DESCRIPTION SYSTEM - VERIFICATION")
    print("="*60 + "\n")
    
    project_root = Path(__file__).parent
    
    # Critical files
    critical_files = [
        "app.py",
        "requirements.txt",
        "README.md",
        "QUICKSTART.md",
    ]
    
    # Backend files
    backend_files = [
        "backend/main.py",
        "backend/requirements.txt",
        "backend/.env.example",
        "backend/app/__init__.py",
        "backend/app/config.py",
        "backend/app/main.py",
        "backend/app/models/llava_model.py",
        "backend/app/models/realviz_model.py",
        "backend/app/pipelines/caption_generator.py",
        "backend/app/pipelines/image_generator.py",
        "backend/app/routes/captions.py",
        "backend/app/routes/images.py",
        "backend/app/schemas/requests.py",
        "backend/app/utils/helpers.py",
    ]
    
    # Frontend files
    frontend_files = [
        "frontend/forensic_app.py",
        "frontend/streamlit_app.py",
        "frontend/requirements.txt",
    ]
    
    print("📋 VERIFYING PROJECT STRUCTURE\n")
    
    # Check critical files
    print("✅ Critical Files:")
    critical_ok = True
    for file in critical_files:
        path = project_root / file
        status = "✓" if path.exists() else "✗"
        print(f"   {status} {file}")
        if not path.exists():
            critical_ok = False
    
    # Check backend
    print("\n✅ Backend Files:")
    backend_ok = True
    for file in backend_files:
        path = project_root / file
        status = "✓" if path.exists() else "✗"
        print(f"   {status} {file}")
        if not path.exists():
            backend_ok = False
    
    # Check frontend
    print("\n✅ Frontend Files:")
    frontend_ok = True
    for file in frontend_files:
        path = project_root / file
        status = "✓" if path.exists() else "✗"
        print(f"   {status} {file}")
        if not path.exists():
            frontend_ok = False
    
    print("\n" + "="*60)
    
    if critical_ok and backend_ok and frontend_ok:
        print("✅ PROJECT STRUCTURE: COMPLETE")
    else:
        print("⚠️  PROJECT STRUCTURE: INCOMPLETE")
    
    print("="*60 + "\n")
    
    return critical_ok and backend_ok and frontend_ok


def check_dependencies():
    """Verify dependencies are installed"""
    print("📦 CHECKING DEPENDENCIES\n")
    
    dependencies = {
        "Core": ["fastapi", "uvicorn", "pydantic", "streamlit", "requests"],
        "Optional (Models)": ["torch", "transformers", "diffusers", "compel"],
        "Utilities": ["pillow"],
    }
    
    all_ok = True
    
    for category, packages in dependencies.items():
        print(f"✅ {category}:")
        for package in packages:
            try:
                __import__(package.replace("-", "_"))
                print(f"   ✓ {package}")
            except ImportError:
                if category == "Optional (Models)":
                    print(f"   ✗ {package} (optional - needed for real image generation)")
                else:
                    print(f"   ✗ {package}")
                    all_ok = False
    
    print("\n" + "="*60)
    
    if all_ok:
        print("✅ DEPENDENCIES: INSTALLED")
    else:
        print("⚠️  DEPENDENCIES: MISSING")
        print("\nInstall missing dependencies with:")
        print("  pip install -r requirements.txt")
        print("  pip install -r backend/requirements.txt")
        print("  pip install -r frontend/requirements.txt")
    
    print("="*60 + "\n")
    
    return all_ok


def check_environment():
    """Check environment configuration"""
    print("⚙️  CHECKING ENVIRONMENT\n")
    
    env_file = Path(__file__).parent / "backend" / ".env"
    
    print("Environment File:")
    if env_file.exists():
        print(f"   ✓ {env_file.name} exists")
        print("\nConfiguration loaded from .env")
    else:
        env_example = Path(__file__).parent / "backend" / ".env.example"
        if env_example.exists():
            print(f"   ✓ .env.example found")
            print(f"   ✗ .env not found (copy from .env.example)")
            print("\nTo configure, run:")
            print(f"   cd backend")
            print(f"   cp .env.example .env")
        else:
            print(f"   ✗ Neither .env nor .env.example found")
    
    print("\n" + "="*60)
    print("✅ ENVIRONMENT: CHECK COMPLETE")
    print("="*60 + "\n")


def print_quick_start():
    """Print quick start instructions"""
    print("🚀 QUICK START\n")
    print("To run the application:\n")
    print("   python app.py\n")
    print("This will start both backend and frontend automatically.\n")
    print("The app will open in your browser at http://localhost:8501\n")
    print("API documentation: http://localhost:8000/docs\n")


def print_next_steps():
    """Print next steps"""
    print("📚 NEXT STEPS\n")
    print("1. Read QUICKSTART.md for immediate setup")
    print("2. Read README.md for complete documentation")
    print("3. Run 'python app.py' to start the system")
    print("4. Test the forensic app in your browser")
    print("5. Check API docs at http://localhost:8000/docs")
    print("6. Enable models when ready (see README.md)\n")


def main():
    """Run all checks"""
    structure_ok = check_structure()
    deps_ok = check_dependencies()
    check_environment()
    
    print("="*60)
    print("✅ VERIFICATION COMPLETE")
    print("="*60 + "\n")
    
    if structure_ok and deps_ok:
        print("🎉 Your system is ready to run!\n")
        print_quick_start()
    else:
        print("⚠️  Please address issues above before running.\n")
        if not deps_ok:
            print("Install dependencies:")
            print("   pip install -r requirements.txt")
            print("   pip install -r backend/requirements.txt")
            print("   pip install -r frontend/requirements.txt\n")
    
    print_next_steps()
    
    print("="*60)
    print("For more help, see documentation files:")
    print("  • QUICKSTART.md")
    print("  • README.md")
    print("  • PROJECT_STRUCTURE.md")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
