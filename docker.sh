Image="test"

sudo docker build -t $Image . 
sudo nvidia-docker run -it $Image