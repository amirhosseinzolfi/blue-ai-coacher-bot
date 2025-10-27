#!/bin/bash

# Copy service file to systemd directory
sudo cp blue_business.service /etc/systemd/system/

# Reload systemd daemon
sudo systemctl daemon-reload

# Enable the service to start on boot
sudo systemctl enable blue_business.service

# Start the service
sudo systemctl start blue_business.service

# Show service status
sudo systemctl status blue_business.service

echo "Blue Business Bot service installed and started!"
echo "Use 'sudo systemctl status blue_business' to check status"
echo "Use 'sudo systemctl restart blue_business' to restart"
echo "Use 'sudo systemctl stop blue_business' to stop"