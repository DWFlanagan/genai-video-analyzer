# GenAI Video Analyzer - Makefile

.PHONY: install setup test lint format clean run help

# Default target
help:
	@echo "GenAI Video Analyzer - Available commands:"
	@echo ""
	@echo "  setup       - Set up the development environment"
	@echo "  install     - Install dependencies with uv"
	@echo "  test        - Run tests"
	@echo "  lint        - Run linting with ruff"
	@echo "  format      - Format code with ruff"
	@echo "  clean       - Clean up generated files"
	@echo "  run         - Run the analyzer (requires VIDEO variable)"
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make setup"
	@echo "  make run VIDEO=path/to/video.mp4"
	@echo "  make test"

# Setup development environment
setup:
	@echo "🔧 Setting up development environment..."
	python3 setup.py

# Install dependencies
install:
	@echo "📦 Installing dependencies with uv..."
	uv sync

# Run tests
test:
	@echo "🧪 Running tests..."
	uv run pytest tests/ -v

# Lint code
lint:
	@echo "🔍 Linting code..."
	uv run ruff check .

# Format code
format:
	@echo "✨ Formatting code..."
	uv run ruff format .

# Clean up
clean:
	@echo "🧹 Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true

# Run analyzer (requires VIDEO variable)
run:
	@if [ -z "$(VIDEO)" ]; then \
		echo "❌ VIDEO variable is required. Usage: make run VIDEO=path/to/video.mp4"; \
		exit 1; \
	fi
	@echo "🎬 Analyzing video: $(VIDEO)"
	uv run video-summarizer "$(VIDEO)" $(ARGS)

# Development mode - watch for changes and run tests
dev:
	@echo "👨‍💻 Development mode - watching for changes..."
	@while true; do \
		make test; \
		echo "Waiting for changes... (Press Ctrl+C to stop)"; \
		sleep 2; \
	done

# Build package
build:
	@echo "📦 Building package..."
	uv build

# Install package locally
install-local:
	@echo "💾 Installing package locally..."
	uv pip install -e .

# Check dependencies
check-deps:
	@echo "🔍 Checking dependencies..."
	uv run python -c "from genai_video_analyzer.utils import check_dependencies; check_dependencies()"
