FROM python:3.7
COPY requirements.txt /requirements.txt
RUN chmod 1777 /tmp

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
WORKDIR /app

# download the scinobo inference graph and the graph embeddings
# RUN wget https://www.dropbox.com/s/24meya731v5ub5d/graph_embeddings_with_L6_21_12_2022.p?dl=0 -O graph_embeddings_with_L6_21_12_2022.p
# RUN wget https://www.dropbox.com/s/qr26k9zbwpeyoyz/scinobo_inference_graph.p?dl=0 -O scinobo_inference_graph.p

COPY . /app

# ENTRYPOINT ["python3", "inference.py"]
# Initialize
CMD ["bash"]
