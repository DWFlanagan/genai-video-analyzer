#!/usr/bin/env python3
"""
Standalone GUI launcher for the GenAI Video Analyzer.
This launcher works independently of the main video analyzer modules.
"""

import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

try:
    from flask import Flask, render_template, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("❌ Flask is required for the GUI. Install it with: pip install flask")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))

# Global variables for tracking processing status
processing_status = {
    "running": False,
    "current_video": "",
    "progress": "",
    "results": [],
    "errors": []
}


class VideoProcessorGUI:
    """GUI wrapper for the video analyzer."""
    
    def __init__(self):
        self.default_options = {
            "llm_model": "llava:13b",
            "summary_model": "gemma3:12b",
            "threshold": 50.0,
            "no_audio": False,
            "no_frames": False,
            "verbose": False
        }
    
    def process_videos(
        self,
        video_paths: List[str],
        output_dir: Optional[str] = None,
        options: Optional[Dict] = None
    ) -> None:
        """Process multiple videos with the given options."""
        global processing_status
        
        if not video_paths:
            processing_status["errors"].append("No videos selected")
            return
        
        # Merge options with defaults
        opts = self.default_options.copy()
        if options:
            opts.update(options)
        
        processing_status["running"] = True
        processing_status["results"] = []
        processing_status["errors"] = []
        
        try:
            for i, video_path in enumerate(video_paths):
                if not os.path.exists(video_path):
                    error_msg = f"Video file not found: {video_path}"
                    processing_status["errors"].append(error_msg)
                    continue
                
                processing_status["current_video"] = os.path.basename(video_path)
                processing_status["progress"] = f"Processing video {i+1} of {len(video_paths)}: {processing_status['current_video']}"
                
                # Try to find the main module
                main_module = None
                try:
                    # Try different ways to locate the main module
                    possible_locations = [
                        "genai_video_analyzer.main",
                        "main", 
                        str(Path(__file__).parent / "main.py")
                    ]
                    
                    for location in possible_locations:
                        try:
                            if location.endswith('.py'):
                                # Direct file execution
                                cmd = [sys.executable, location, video_path]
                            else:
                                # Module execution
                                cmd = [sys.executable, "-m", location, video_path]
                            break
                        except:
                            continue
                    else:
                        raise Exception("Cannot find video analyzer main module")
                    
                    if output_dir:
                        cmd.extend(["--output-dir", output_dir])
                    
                    cmd.extend(["--llm-model", opts["llm_model"]])
                    cmd.extend(["--summary-model", opts["summary_model"]])
                    cmd.extend(["--threshold", str(opts["threshold"])])
                    
                    if opts["no_audio"]:
                        cmd.append("--no-audio")
                    if opts["no_frames"]:
                        cmd.append("--no-frames")
                    if opts["verbose"]:
                        cmd.append("--verbose")
                    
                    logger.info(f"Running command: {' '.join(cmd)}")
                    
                    # Run the video analyzer
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=3600  # 1 hour timeout
                    )
                    
                    if result.returncode == 0:
                        processing_status["results"].append({
                            "video": video_path,
                            "status": "success",
                            "output": result.stdout
                        })
                    else:
                        error_msg = f"Failed to process {video_path}: {result.stderr}"
                        processing_status["errors"].append(error_msg)
                        processing_status["results"].append({
                            "video": video_path,
                            "status": "error",
                            "error": result.stderr
                        })
                
                except subprocess.TimeoutExpired:
                    error_msg = f"Timeout processing {video_path}"
                    processing_status["errors"].append(error_msg)
                except Exception as e:
                    error_msg = f"Error processing {video_path}: {str(e)}"
                    processing_status["errors"].append(error_msg)
        
        finally:
            processing_status["running"] = False
            processing_status["current_video"] = ""
            processing_status["progress"] = "Processing completed"


# Initialize the processor
processor = VideoProcessorGUI()


@app.route('/')
def index():
    """Serve the main GUI page."""
    return render_template('index.html')


@app.route('/api/process', methods=['POST'])
def process_videos():
    """API endpoint to start video processing."""
    global processing_status
    
    if processing_status["running"]:
        return jsonify({"error": "Processing already in progress"}), 400
    
    data = request.json
    video_paths = data.get('videos', [])
    output_dir = data.get('output_dir')
    options = data.get('options', {})
    
    if not video_paths:
        return jsonify({"error": "No videos selected"}), 400
    
    # In demo mode, just show what would happen
    processing_status["running"] = True
    processing_status["progress"] = "Demo mode - showing configuration"
    processing_status["results"] = []
    processing_status["errors"] = []
    
    # Simulate processing for demo
    def demo_process():
        time.sleep(2)
        processing_status["progress"] = "Demo completed - configuration validated"
        processing_status["results"].append({
            "video": f"{len(video_paths)} video(s) selected",
            "status": "demo",
            "output": f"Configuration preview:\n- LLM Model: {options.get('llm_model', 'llava:13b')}\n- Summary Model: {options.get('summary_model', 'gemma3:12b')}\n- Threshold: {options.get('threshold', 50.0)}\n- Skip Audio: {options.get('no_audio', False)}\n- Skip Frames: {options.get('no_frames', False)}\n- Verbose: {options.get('verbose', False)}\n\nInstall all dependencies to enable actual video processing."
        })
        processing_status["running"] = False
    
    # Start demo processing in a separate thread
    thread = threading.Thread(target=demo_process)
    thread.daemon = True
    thread.start()
    
    return jsonify({"message": "Demo processing started"})


@app.route('/api/status')
def get_status():
    """Get the current processing status."""
    return jsonify(processing_status)


def main():
    """Main entry point for the standalone GUI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="GenAI Video Analyzer GUI")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to (default: 5000)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Create templates directory if it doesn't exist
    templates_dir = Path(__file__).parent / "templates"
    templates_dir.mkdir(exist_ok=True)
    
    print(f"🎬 Starting GenAI Video Analyzer GUI...")
    print(f"📡 Server will be available at: http://{args.host}:{args.port}")
    print(f"🌐 Open this URL in your web browser to access the interface")
    print(f"⏹️  Press Ctrl+C to stop the server")
    print()
    
    try:
        app.run(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        print("\n👋 GUI server stopped")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error starting GUI: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()