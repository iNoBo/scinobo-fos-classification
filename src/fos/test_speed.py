""" 

This script tests the speed of the algorithm and finds the bottleneck.

"""

import json
import time
from inference import infer, create_payload


# test input json
my_path = "/mnt/data/test_fos_data/102_chunk.json"

def load_json(path):
    """Load json file."""
    with open(path, 'r') as f:
        data = json.load(f)
    return data


if __name__ == '__main__':
    # load data -- use the first 500 entries
    data = load_json(my_path)[:500]
    # load mappings
    with open('L2_to_L1.json', 'r') as fin:
        L2_to_L1 = json.load(fin)
    with open('L3_to_L2.json', 'r') as fin:
        L3_to_L2 = json.load(fin)
    with open('L4_to_L3.json', 'r') as fin:
        L4_to_L3 = json.load(fin)
    # time the create_payload function
    start_create_payload = time.time()
    payload = create_payload(data)
    end_create_payload = time.time()
    print("Create payload time: {}".format(end_create_payload - start_create_payload))
    # time the infer function
    infer_start = time.time()
    # the other functions will log their time in their inference logger
    my_res = infer(payload=payload, only_l4=True)
    infer_end = time.time()
    print("Infer time: {}".format(infer_end - infer_start))
    # save the predictions
    with open('/mnt/data/test_fos_data/test_predictions_optimized.json', 'w') as fout:
        json.dump(my_res, fout)