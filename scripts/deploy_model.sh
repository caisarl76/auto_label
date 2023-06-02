docker rm -f mnc-minit-autolabeling
docker run -d -p 10051:80 --name mnc-minit-autolabeling mnc/minit-autolabeling:latest