FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

RUN chmod 1777 /tmp

WORKDIR /app/src/fos/data/

# Install wget
RUN apt-get update && apt-get install -y wget

# handle the large files for the inference graph
ARG HF_TOKEN
# download the scinobo inference graph and the graph embeddings from HF organization
RUN wget https://huggingface.co/datasets/iNoBo/scinobo-fos-graph-embeddings/resolve/main/graph_embeddings_with_L6_21_12_2022.p?download=true -O graph_embeddings_with_L6_21_12_2022.p
RUN wget https://huggingface.co/datasets/iNoBo/scinobo-fos-inference-graph/resolve/main/scinobo_inference_graph.json?download=true -O scinobo_inference_graph.json

WORKDIR /app

# Copy only the requirements file, to cache the installation of dependencies
COPY requirements.txt /app/requirements.txt

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
EXPOSE 7860

# Copy the rest of your application
COPY . /app

# Set the PYTHONPATH to include the src directory
ENV PYTHONPATH "${PYTHONPATH}:/app/src"

# Change the working directory to where the Gradio app is located
WORKDIR /app/src/fos/server

# Expose the port that Gradio uses
EXPOSE 7860

# Run the Gradio app
CMD ["python3", "gradio_app.py"]
