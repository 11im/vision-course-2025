# Use the official PaddlePaddle image as the base
FROM paddlepaddle/paddle:3.2.0-gpu-cuda12.6-cudnn9.5

# Set the working directory inside the container
WORKDIR /work

# Install system-level dependencies required by OpenCV and other libraries
# Using noninteractive frontend to avoid prompts during build
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the project files into the container
# .dockerignore will prevent logs, results, data, etc. from being copied
COPY . .

# Install Python dependencies with specific versions for compatibility
RUN pip install --no-cache-dir \
    paddleocr==2.7.0.3 \
    sentence-transformers \
    numpy==1.23.5

# Set the default command to execute when the container starts
# This will automatically run the experiment script
CMD ["bash", "run.sh"]
