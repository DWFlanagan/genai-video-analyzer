#!/usr/bin/env python3
"""
Web GUI for the GenAI Video Analyzer.

Provides a simple web interface for selecting videos and configuring
analysis options without using the command line.
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

from flask import Flask, render_template, request, jsonify, send_from_directory

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

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
                
                # Build command line arguments
                cmd = [sys.executable, "-m", "genai_video_analyzer.main", video_path]
                
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
                
                try:
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
    
    # Start processing in a separate thread
    thread = threading.Thread(
        target=processor.process_videos,
        args=(video_paths, output_dir, options)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({"message": "Processing started"})


@app.route('/api/status')
def get_status():
    """Get the current processing status."""
    return jsonify(processing_status)


@app.route('/api/files')
def list_files():
    """List video files in a directory."""
    directory = request.args.get('dir', os.getcwd())
    
    try:
        if not os.path.exists(directory):
            return jsonify({"error": "Directory not found"}), 404
        
        files = []
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isfile(item_path):
                # Check if it's likely a video file
                ext = os.path.splitext(item)[1].lower()
                if ext in ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm']:
                    files.append({
                        "name": item,
                        "path": item_path,
                        "size": os.path.getsize(item_path)
                    })
        
        # Also return parent directory for navigation
        parent_dir = os.path.dirname(directory) if directory != os.path.dirname(directory) else None
        
        return jsonify({
            "directory": directory,
            "parent": parent_dir,
            "files": files
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/directories')
def list_directories():
    """List directories for browsing."""
    directory = request.args.get('dir', os.getcwd())
    
    try:
        if not os.path.exists(directory):
            return jsonify({"error": "Directory not found"}), 404
        
        dirs = []
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                dirs.append({
                    "name": item,
                    "path": item_path
                })
        
        parent_dir = os.path.dirname(directory) if directory != os.path.dirname(directory) else None
        
        return jsonify({
            "directory": directory,
            "parent": parent_dir,
            "directories": dirs
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def create_gui_app():
    """Create and configure the Flask app."""
    # Create templates directory if it doesn't exist
    templates_dir = Path(__file__).parent / "templates"
    templates_dir.mkdir(exist_ok=True)
    
    # Create static directory if it doesn't exist  
    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(exist_ok=True)
    
    return app


def run_gui(host='127.0.0.1', port=5000, debug=False):
    """Run the GUI server."""
    logger.info(f"Starting GenAI Video Analyzer GUI at http://{host}:{port}")
    logger.info("Open the URL in your web browser to access the interface")
    
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GenAI Video Analyzer GUI")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    create_gui_app()
    run_gui(host=args.host, port=args.port, debug=args.debug)