#!/usr/bin/env python3
"""
GUI launcher for the GenAI Video Analyzer.
"""

import sys
from pathlib import Path

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="GenAI Video Analyzer GUI")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to (default: 5000)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    print(f"🎬 Starting GenAI Video Analyzer GUI...")
    print(f"📡 Server will be available at: http://{args.host}:{args.port}")
    print(f"🌐 Open this URL in your web browser to access the interface")
    print(f"⏹️  Press Ctrl+C to stop the server")
    print()
    
    try:
        # Import the GUI module
        from . import gui
        gui.create_gui_app()
        gui.run_gui(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        print("\n👋 GUI server stopped")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error starting GUI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()