FROM continuumio/miniconda3:24.1.2-0

RUN chmod 1777 /tmp

WORKDIR /app/src/fos/data/

# handle the large files for the inference graph
ARG HF_TOKEN
# download the scinobo inference graph and the graph embeddings from HF organization
RUN wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/datasets/iNoBo/scinobo-fos-graph-embeddings/resolve/main/graph_embeddings_with_L6_21_12_2022.p?download=true -O graph_embeddings_with_L6_21_12_2022.p
RUN wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/datasets/iNoBo/scinobo-fos-inference-graph/resolve/main/scinobo_inference_graph.p?download=true -O scinobo_inference_graph.p

WORKDIR /app

# Copy only the requirements file, to cache the installation of dependencies
COPY requirements.txt /app/requirements.txt

# Create a Conda environment
RUN conda create -n docker_env python=3.11 -y

# Activate the Conda environment
SHELL ["conda", "run", "-n", "docker_env", "/bin/bash", "-c"]

# Install PyTorch with CUDA support
# Adjust the PyTorch and CUDA versions as needed
# Check https://pytorch.org/get-started/locally/ for the correct command for your needs
RUN conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# COPY DESCRIPTIONS
# install dependencies
RUN python3 -m pip install -r requirements.txt

# download spacy model
RUN python3 -m spacy download en_core_web_sm

# download nltk punkt and stopwords
RUN python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
# RUN python3 -m nltk.downloader stopwords

# Build directory architecture
RUN mkdir /input_files
RUN mkdir /output_files

# Expose the port the app runs on
EXPOSE 1997

# change the working directory
WORKDIR /src

COPY ./src .

# ENTRYPOINT ["python3", "inference.py"]
# Initialize
CMD ["bash"]
