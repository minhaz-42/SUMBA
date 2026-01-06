# SUMBA Development Makefile

.PHONY: help install dev migrate run test lint clean docker-build docker-run train

# Default target
help:
	@echo "SUMBA - 3D Sign Language Translation System"
	@echo ""
	@echo "Usage:"
	@echo "  make install     Install Python dependencies"
	@echo "  make dev         Install development dependencies"
	@echo "  make migrate     Run database migrations"
	@echo "  make run         Start development server"
	@echo "  make test        Run tests"
	@echo "  make lint        Run linting"
	@echo "  make clean       Clean build artifacts"
	@echo "  make train       Start model training"
	@echo ""

# Install dependencies
install:
	pip install -r requirements.txt

# Install development dependencies
dev:
	pip install -r requirements.txt
	pip install pytest pytest-django pytest-cov black isort flake8 mypy

# Database migrations
migrate:
	python manage.py makemigrations
	python manage.py migrate

# Run development server
run:
	python manage.py runserver

# Run with ASGI (WebSocket support)
run-asgi:
	daphne -b 0.0.0.0 -p 8000 core.asgi:application

# Run Redis server
redis:
	redis-server

# Run all services (requires tmux)
run-all:
	tmux new-session -d -s sumba 'make redis'
	tmux split-window -h 'make run-asgi'
	tmux attach-session -t sumba

# Run tests
test:
	pytest --cov=backend --cov-report=html

# Run linting
lint:
	black --check backend/ ml/
	isort --check-only backend/ ml/
	flake8 backend/ ml/

# Format code
format:
	black backend/ ml/
	isort backend/ ml/

# Type checking
typecheck:
	mypy backend/ ml/

# Clean build artifacts
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .mypy_cache
	rm -rf dist
	rm -rf *.egg-info

# Create superuser
superuser:
	python manage.py createsuperuser

# Collect static files
collectstatic:
	python manage.py collectstatic --noinput

# Docker commands
docker-build:
	docker-compose build

docker-run:
	docker-compose up

docker-down:
	docker-compose down

# Training commands
train:
	python ml/train.py \
		--data_dir data/ASL \
		--model_type hybrid \
		--batch_size 32 \
		--epochs 50 \
		--output_dir checkpoints/hybrid_asl

train-stgcn:
	python ml/train.py \
		--data_dir data/ASL \
		--model_type stgcn \
		--batch_size 32 \
		--epochs 50 \
		--output_dir checkpoints/stgcn_asl

train-transformer:
	python ml/train.py \
		--data_dir data/ASL \
		--model_type transformer \
		--batch_size 32 \
		--epochs 50 \
		--output_dir checkpoints/transformer_asl

# Evaluation
evaluate:
	python ml/train.py \
		--data_dir data/ASL \
		--checkpoint checkpoints/hybrid_asl/best.pt \
		--evaluate_only

# Export model for deployment
export-model:
	python ml/export.py \
		--checkpoint checkpoints/hybrid_asl/best.pt \
		--output models/hybrid_asl.onnx

# Generate API documentation
docs:
	python manage.py generateschema --file openapi.yaml

# Create new app
app:
	@read -p "App name: " name; \
	python manage.py startapp $$name backend/$$name
