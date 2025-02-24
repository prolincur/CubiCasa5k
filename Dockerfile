# "docker build -t cubi -f Dockerfile ." # '-f is used when the docker file is not called "Dockerfile" but something else. In this case it is reduntant. But good to know.'
# Run without volume "docker run -d --rm --init --gpus all --ipc=host --publish 1111:1111 -e NVIDIA_VISIBLE_DEVICES=0 cubi_localonly:0.1.0"

FROM anibali/pytorch:cuda-9.0

RUN sudo apt-get update && \
    sudo apt-get upgrade -y && \
    sudo apt-get install -y \
        build-essential 

# Copy the requirements.txt first to leverage Docker cache
COPY requirements.txt /app/

WORKDIR /app

RUN pip install -r requirements.txt

# Copy your application code into the container image
COPY . /app

# Expose port 1111 for the Flask app to run on
EXPOSE 1111

# Command to run the application
CMD ["python", "cubicasa_detection.py"]

# Older commands to run the jupyter notebook and to run the script with "--volume" header on. Change the file name accordingly. 
# To run Jupyter-lab server "docker run --rm -it --init --gpus all --ipc=host --publish 1111:1111 --volume="C:/Users/Pranay/Desktop/Prolincur_codes/CubiCasa5k:/app" -e NVIDIA_VISIBLE_DEVICES=0 cubi jupyter-lab --port 1111 --ip 0.0.0.0 --no-browser"
# To run cubicasa_detection.py "docker run -d --rm --init --gpus all --ipc=host --publish 1111:1111 --volume="C:/Users/Pranay/Desktop/Prolincur_codes/CubiCasa5k:/app" -e NVIDIA_VISIBLE_DEVICES=0 cubi python /app/cubicasa_detection.py"
