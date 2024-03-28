FROM continuumio/miniconda3:24.1.2-0

RUN chmod 1777 /tmp

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
RUN pip3 install -r requirements.txt

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

# download the scinobo inference graph and the graph embeddings from HF organization
# for passing secret arguments
# https://huggingface.co/docs/hub/en/spaces-sdks-docker
RUN --mount=type=secret,id=sotkot_hf_token,mode=0444,required=true \
	wget --header="Authorization: Bearer $(cat /run/secrets/sotkot_hf_token)" https://huggingface.co/datasets/iNoBo/scinobo-fos-graph-embeddings/resolve/main/graph_embeddings_with_L6_21_12_2022.p?download=true -O graph_embeddings_with_L6_21_12_2022.p
RUN --mount=type=secret,id=sotkot_hf_token,mode=0444,required=true \
	wget --header="Authorization: Bearer $(cat /run/secrets/sotkot_hf_token)" https://huggingface.co/datasets/iNoBo/scinobo-fos-inference-graph/resolve/main/scinobo_inference_graph.p?download=true -O scinobo_inference_graph.p

COPY . /app

# ENTRYPOINT ["python3", "inference.py"]
# Initialize
CMD ["bash"]
