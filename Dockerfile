FROM jupyter/base-notebook:latest
WORKDIR /home/jovyan/work

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

COPY . /home/jovyan/work

EXPOSE 8888

CMD ["start-notebook.py","--NotebookApp.token=","--NotebookApp.password=","--NotebookApp.allow_origin=*","--NotebookApp.ip=0.0.0.0","--NotebookApp.port=8888","--NotebookApp.open_browser=False"]

# Commands to build and run the Docker container:
# podman build --format docker -t intro-ds -f Dockerfile .
# podman tag intro-ds docker.io/jocowhite/intro-ds:latest
# podman push docker.io/jocowhite/intro-ds:latest