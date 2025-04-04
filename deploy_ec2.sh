#!/bin/bash

# Exit on error
set -e

echo "Starting deployment script..."

# Update system packages
echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker
echo "Installing Docker..."
sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
sudo apt-get install -y docker-ce

# Start Docker service
echo "Starting Docker service..."
sudo systemctl start docker
sudo systemctl enable docker

# Add current user to docker group
echo "Adding current user to docker group..."
sudo usermod -aG docker ${USER}

# Install Docker Compose
echo "Installing Docker Compose..."
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.3/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Create project directory
echo "Creating project directory..."
mkdir -p ~/bank-marketing-prediction
cd ~/bank-marketing-prediction

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p data models static/images logs

# Clone the repository if it doesn't exist
if [ ! -d .git ]; then
    echo "Cloning the repository..."
    git init
    git remote add origin https://github.com/yourusername/bank-marketing-prediction.git
    git pull origin main
else
    echo "Repository already exists, pulling latest changes..."
    git pull origin main
fi

# Build and start the Docker container
echo "Building and starting Docker containers..."
sudo docker-compose up -d --build

# Wait for the application to start
echo "Waiting for the application to start..."
sleep 10

# Check if the application is running
echo "Checking if the application is running..."
if curl -s http://localhost/health | grep -q "healthy"; then
    echo "Deployment completed successfully! Application is running."
else
    echo "Deployment failed. Application is not running properly."
    echo "Checking Docker logs..."
    sudo docker-compose logs
    exit 1
fi

echo "Deployment completed successfully!"