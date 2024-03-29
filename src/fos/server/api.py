""" 

FastAPI for the FoS classifier. This docstring will be updated.

"""

import logging
import traceback

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from fos.server.logging_setup import setup_root_logger
from fos.pipeline.inference import create_payload, infer, process_pred

# init the logger
setup_root_logger()
LOGGER = logging.getLogger(__name__)
LOGGER.info("FoS Api initialized")

# declare classes for input-output and error responses
class FoSInferRequest(BaseModel):
    # based on the request config, the request data should contain the following fields
    id: str
    title: str | None = ""
    abstract: str | None = ""
    pub_venue: str | None = ""
    cit_venues: list[str] | None = []
    ref_venues: list[str] | None = []
    # add an example for the request data
    model_config = {
        "id": "10.18653/v1/w19-5032",
        "title": "Embedding Biomedical Ontologies by Jointly Encoding Network Structure and Textual Node Descriptors",
        "abstract": "Network Embedding (NE) methods, which map network nodes to low-dimensional feature vectors, have wide applications in network analysis and bioinformatics. Many existing NE methods rely only on network structure, overlooking other information associated with the nodes, e.g., text describing the nodes. Recent attempts to combine the two sources of information only consider local network structure. We extend NODE2VEC, a well-known NE method that considers broader network structure, to also consider textual node descriptors using recurrent neural encoders. Our method is evaluated on link prediction in two networks derived from UMLS. Experimental results demonstrate the effectiveness of the proposed approach compared to previous work.",
        "pub_venue": "proceedings of the bionlp workshop and shared task",
        "cit_venues": [
            "acl",
            "acl",
            "aimag",
            "arxiv artificial intelligence",
            "arxiv computation and language",
            "arxiv machine learning",
            "arxiv social and information networks",
            "briefings in bioinformatics",
            "comparative and functional genomics",
            "conference of the european chapter of the association for computational linguistics",
            "cvpr",
            "emnlp",
            "emnlp",
            "emnlp",
            "emnlp",
            "emnlp",
            "emnlp",
            "emnlp",
            "eswc",
            "iclr",
            "icml",
            "ieee trans signal process",
            "j mach learn res",
            "kdd",
            "kdd",
            "kdd",
            "kdd"
        ],
        "ref_venues": [
           "naacl",
            "nips",
            "nucleic acids res",
            "pacific symposium on biocomputing",
            "physica a statistical mechanics and its applications",
            "proceedings of the acm conference on bioinformatics computational biology and health informatics",
            "sci china ser f",
            "the web conference"
        ]
    }
    
    
class FoSInferRequestResponse(BaseModel):
    # based on the request config, the response data should contain the following fields
    id: str
    L1: str | None = "N/A"
    L2: str | None = "N/A"
    L3: str | None = "N/A"
    L4: str | None = "N/A"
    L5: str | None = "N/A"
    L6: str | None = "N/A"
    score_for_L3: float | None = 0.0
    score_for_L4: float | None = 0.0
    # add an example for the response data
    model_config = {
        "id": "the id of the request",
        "L1": "the L1 FoS pred",
        "L2": "the L2 FoS pred",
        "L3": "the L3 FoS pred",
        "L4": "the L4 FoS pred",
        "L5": "the L5 FoS pred", 
        "L6": "the L6 FoS pred",
        "score_for_L3": "the score for L3",
        "score_for_L4": "the score for L4"
    }
    
    
class FoSInferRequests(BaseModel):
    # based on the request config, the request data should contain the following fields
    data: list[FoSInferRequest]


class FoSInferRequestsResponse(BaseModel):
    # based on the request config, the response data should contain the following fields
    data: list[FoSInferRequestResponse]


class ErrorResponse(BaseModel):
    success: int
    message: str


# the FastAPI app
app = FastAPI()


# handle CORS -- at a later stage we can restrict the origins
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# create a middleware that logs the requests -- this function logs everything. It might not be needed.
@app.middleware("http")
async def log_requests(request, call_next):
    LOGGER.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    return response


# create an endpoint which receives a request and just returns the request data
@app.post("/echo", response_model=FoSInferRequests, responses={400: {"model": ErrorResponse}})
def echo(request_data: FoSInferRequests):
    LOGGER.info(f"Request data for echo: {request_data}")
    return request_data.model_dump() 


@app.post("/infer_fos", response_model=FoSInferRequestsResponse, responses={400: {"model": ErrorResponse}})
def infer_publications(request_data: FoSInferRequests):
    # TODO: update the docstring, to be more informative.
    """
    Infer the field of science of a publication based on the publication's metadata. 

    Args:
        request_data (FoSInferRequests): The request data containing the publication's metadata.

    Returns:
        FoSInferRequestResponse: The response data containing the publication's metadata and the inferred field of study.
    """
    LOGGER.info(f"Request data: {request_data}") # this is formatted based on the BaseModel classes
    try:
        # process the input data to convert it to json -- since we are here, the format is OK. This is handled by FastAPI
        request_data = request_data.model_dump() 
        # create the payload
        payload = create_payload(request_data['data'])
        # infer the FoS
        preds = infer(payload = payload)
        # process the predictions
        response_data = process_pred(preds)
        response_data = FoSInferRequestsResponse(data=response_data)
        LOGGER.info(f"Response data: {response_data}") # this is formatted based on the BaseModel classes
        return response_data
    except Exception as e:
        LOGGER.error(f"Error: {e}")
        return HTTPException(status_code=400, detail={"success": 0, "message": f"{str(e)}\n{traceback.format_exc()}"})
    
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1997)