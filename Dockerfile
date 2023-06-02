FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

ENV \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Seoul \
    LC_ALL=C.UTF-8 \
    UWSGI_NPROC=1

RUN \
    conda install -c conda-forge uwsgi && \
    ln -s /opt/conda/bin/uwsgi /usr/local/bin/uwsgi && \
    apt-get update && apt-get install -y --no-install-recommends \
        curl \
        vim \
        nginx \
        supervisor \
        libgl1-mesa-glx \
        tzdata \
        libglib2.0-0 \
    && \
    apt-get -o Dpkg::Options::='--force-confmiss' install --reinstall -y netbase

# Install Python packages
COPY /requirements.txt /tmp/requirements.txt
RUN \
    pip install --upgrade pip setuptools && \
    pip install -r /tmp/requirements.txt

# Clean up
RUN \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /root/.cache && \
    rm -rf /var/lib/apt/lists/*

# Copy app files
COPY . /
RUN \
    rm /requirements.txt /Dockerfile && \
    ln -sf /configs/nginx.conf /etc/nginx/conf.d/server.conf && \
    ln -sf /configs/supervisor.conf /etc/supervisor/conf.d/programs.conf

EXPOSE 80
CMD ["supervisord", "-n"]