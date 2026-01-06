#!/usr/bin/env python3
"""
Wrapper script to run Django commands from the Sumba root directory.
Delegates to signlang3d/backend/manage.py
"""
import os
import sys
import subprocess

def main():
    # Get the directory where this script is located
    root_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(root_dir, 'signlang3d', 'backend')
    venv_python = os.path.join(root_dir, 'signlang3d', 'venv', 'bin', 'python')
    manage_py = os.path.join(backend_dir, 'manage.py')
    
    # Check if venv exists
    if not os.path.exists(venv_python):
        print(f"Error: Virtual environment not found at {venv_python}")
        print("Please create it first: cd signlang3d && python3 -m venv venv")
        sys.exit(1)
    
    # Set environment variable for SQLite
    env = os.environ.copy()
    env['USE_SQLITE'] = 'true'
    
    # Run the actual manage.py with all arguments
    cmd = [venv_python, manage_py] + sys.argv[1:]
    
    try:
        result = subprocess.run(cmd, cwd=backend_dir, env=env)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        sys.exit(0)

if __name__ == '__main__':
    main()
