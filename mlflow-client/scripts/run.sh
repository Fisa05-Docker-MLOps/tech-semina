#!/bin/bash

# run the Docker container with GPU support
# execute the script in the top level directory
docker run --gpus all -d \
  -v "$(pwd)":/workspace \
  -w /workspace \
  ljh4770/torch-image:base \
  sleep infinity