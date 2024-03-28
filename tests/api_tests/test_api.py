""" 

This test script will be used to test the API endpoints for the Field of Science (FoS) inference service.
We will test the following:
    - the API can accept requests 
    - the API can return responses
    - the API can handle errors
    - the API can handle missing data
    - the API can handle invalid data
    - the API can handle invalid requests
    - the API can handle invalid responses
"""

from fastapi.testclient import TestClient
from fos.api import app


client = TestClient(app)


def test_infer_publications():
    # Test case for successful inference of publications
    request_data = {
        "data": [
            {
                "id": "1",
                "title": "Publication 1",
                "abstract": "Abstract 1",
                "pub_venue": "Venue 1",
                "cit_venues": ["Venue 2", "Venue 3"],
                "ref_venues": ["Venue 4", "Venue 5"]
            },
            {
                "id": "2",
                "title": "Publication 2",
                "abstract": "Abstract 2",
                "pub_venue": "Venue 6",
                "cit_venues": ["Venue 7", "Venue 8"],
                "ref_venues": ["Venue 9", "Venue 10"]
            }
        ]
    }
    response = client.post("/infer_fos", json=request_data)
    assert response.status_code == 200
    assert len(response.json()["data"]) == 2
    assert response.json()["data"][0]["id"] == "1"
    assert response.json()["data"][1]["id"] == "2"
    assert response.json()["data"][0]["L1"] == "N/A"
    assert response.json()["data"][0]["L2"] == "N/A"
    assert response.json()["data"][0]["L3"] == "N/A"
    assert response.json()["data"][0]["L4"] == "N/A"
    assert response.json()["data"][0]["L5"] == "N/A"
    assert response.json()["data"][0]["L6"] == "N/A"
    assert response.json()["data"][0]["score_for_L3"] == 0.0
    assert response.json()["data"][0]["score_for_L4"] == 0.0


def test_infer_publications_invalid_request():
    # Test case for invalid request data
    request_data = {
        "data": [
            {
                "id": "1",
                "title": "Publication 1",
                "abstract": "Abstract 1",
                "pub_venue": "Venue 1",
                "cit_venues": ["Venue 2", "Venue 3"],
                "ref_venues": ["Venue 4", "Venue 5"]
            },
            {
                "id": "2",
                "title": "Publication 2",
                "abstract": "Abstract 2",
                "pub_venue": "Venue 6",
                "cit_venues": ["Venue 7", "Venue 8"],
                "ref_venues": ["Venue 9", "Venue 10"]
            }
        ]
    }
    # Remove the required field 'title' from the first publication
    del request_data["data"][0]["id"]
    response = client.post("/infer_fos", json=request_data)
    assert response.status_code == 400
    assert response.json()["success"] == 0
    assert "message" in response.json()


def test_infer_publications_empty_data():
    # Test case for empty request data
    request_data = {
        "data": []
    }
    response = client.post("/infer_fos", json=request_data)
    assert response.status_code == 200
    assert len(response.json()["data"]) == 1
    assert response.json()["data"][0]["L1"] == "N/A"
    assert response.json()["data"][0]["L2"] == "N/A"
    assert response.json()["data"][0]["L3"] == "N/A"
    assert response.json()["data"][0]["L4"] == "N/A"
    assert response.json()["data"][0]["L5"] == "N/A"
    assert response.json()["data"][0]["L6"] == "N/A"
    assert response.json()["data"][0]["score_for_L3"] == 0.0
    assert response.json()["data"][0]["score_for_L4"] == 0.0


def test_publications_missing_data():
    # Test case for missing fields in request data -- we check if the API can handle missing fields
    # by using default values
    request_data = {
        "data": [
            {
                "id": "1",
                "title": "Publication 1",
                "pub_venue": "Venue 1"
            }
        ]
    }
    response = client.post("/echo", json=request_data)
    assert response.status_code == 200
    assert len(response["data"]) == 1
    assert response.json()["data"][0]["id"] == "1"
    assert response.json()["data"][0]["title"] == "Publication 1"
    assert response.json()["data"][0]["abstract"] == ""
    assert response.json()["data"][0]["pub_venue"] == "Venue 1"
    assert response.json()["data"][0]["cit_venues"] == []
    assert response.json()["data"][0]["ref_venues"] == []
    
    
def test_infer_publications_missing_data():
    # Test case for missing fields in request data
    request_data = {
        "data": [
            {
                "id": "1",
                "title": "Publication 1",
                "abstract": "Abstract 1",
                "pub_venue": "Venue 1"
            }
        ]
    }
    response = client.post("/infer_fos", json=request_data)
    assert response.status_code == 200
    assert len(response.json()["data"]) == 1
    assert response.json()["data"][0]["L1"] == "N/A"
    assert response.json()["data"][0]["L2"] == "N/A"
    assert response.json()["data"][0]["L3"] == "N/A"
    assert response.json()["data"][0]["L4"] == "N/A"
    assert response.json()["data"][0]["L5"] == "N/A"
    assert response.json()["data"][0]["L6"] == "N/A"
    assert response.json()["data"][0]["score_for_L3"] == 0.0
    assert response.json()["data"][0]["score_for_L4"] == 0.0
