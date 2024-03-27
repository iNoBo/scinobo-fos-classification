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

# download the scinobo inference graph and the graph embeddings from HF organization
# for passing secret arguments
# https://huggingface.co/docs/hub/en/spaces-sdks-docker
RUN --mount=type=secret,id=sotkot_hf_token,mode=0444,required=true \
	wget --header="Authorization: Bearer $(cat /run/secrets/sotkot_hf_token)" <the URL> -O graph_embeddings_with_L6_21_12_2022.p
RUN --mount=type=secret,id=sotkot_hf_token,mode=0444,required=true \
	wget --header="Authorization: Bearer $(cat /run/secrets/sotkot_hf_token)" <the URL> -O scinobo_inference_graph.p

COPY . /app

# ENTRYPOINT ["python3", "inference.py"]
# Initialize
CMD ["bash"]
