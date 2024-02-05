# scinobo-fos-classification
This repository contains the code and the dockerfile for building the image and container responsible for the Field of Science classification of scientific publications.

## Related publications
- Nikolaos Gialitsis, Sotiris Kotitsas, and Haris Papageorgiou. 2022. SciNoBo: A Hierarchical Multi-Label Classifier of Scientific Publications. In Companion Proceedings of the Web Conference 2022 (WWW '22). Association for Computing Machinery, New York, NY, USA, 800–809. https://doi.org/10.1145/3487553.3524677
- Kotitsas S, Pappas D, Manola N, Papageorgiou H. SCINOBO: a novel system classifying scholarly communication in a dynamically constructed hierarchical Field-of-Science taxonomy. Front Res Metr Anal. 2023 May 4;8:1149834. doi: 10.3389/frma.2023.1149834. PMID: 37215249; PMCID: PMC10192702.

## Contents of repository
- input_files: Directory which contains test files for a demo. If you want to test the docker, then use this folder as the input volume to the docker when you run it. E.g. -v path/to/input_files:
- Dockerfile: Contains the commands for building the docker
- inference.py: Contains the code responsible for the inference procedure. This is the main script.
- input_schema.json: Example schema for how the input should be.
- L2_to_L1.json: Mapping from the L2 FoS fields to L1
- L3_to_L2.json: Mapping from the L3 FoS fields to L2
- L3_to_L4.json: Mapping from the L3 FoS fields to L4
- L4_to_L3.json: Mapping from the L4 FoS fields to L3
- multigraph.py: Contains the code for managing the inference graph of SciNoBo
- output_schema.json: Contains the output schema of the predictions
- requirements.txt: The python packages required.
- scinobo_inference_graph.p: The SciNoBo inference graph
- utils.py: Contains code for utilities
- venue_parser.py: Contains code for parsing the venue names
- venue_maps.p: Contains the abbreviations of the venues

## Commands to build and run the docker

## Create docker image
Use the following command to create a docker image.
The flag -t specifies the name of the image that will be created with an optional tag (for example its version).
`docker build <-t NAME:tag> <Dockerfile location>`

## Example
`docker build --tag intelcomp_fos .`

- The name of the image in this case is intelcomp_fos, with no specific version.
- The location of the Dockerfile is the current directory.

## Run image container
To run a container with the previous configuration, the following command is needed:

`docker run <--rm> -i <--name CONTAINER-NAME> -v path/to/input/local/data:/input_files -v path/to/output/local/data:/output_files IMAGE-NAME python inference.py <args>`

## Flags:

- --rm: remove the container when execution ends. (optional)
- -i: set interactive mode. (optional)
- --name: a name for the container. (optional)
- v: volume binding. It maps a local directory to a directory inside the container so that local files can be accessed from it. The format is: /absolute/path/to/local/dir:/absolute/path/to/container/dir. You need to also specify a local directory where the docker will save the output

**Reminder**: If you want to test the docker, you need to use the following path as an input path --> -v path/to/input_files

## Example

`docker run --rm -i --name fos_inference_docker -v path/to/input/local/data/:/input_files -v path/to/output/local/data:/output_files intelcomp_fos python inference.py --file_type="parquet"`

- **path/to/input/local/data/** --> where the input files exist in the host
- **path/to/output/local/data** --> a directory in the host where you want the container to save the output

## Test example

`docker run --rm -i --name fos_inference_docker -v /path/to/input_files/demo_parquet:/input_files -v path/to/output/local/data:/output_files intelcomp_fos python inference.py --file_type="parquet"`

## Miscellaneous
- **If you want to connect to the docker -->** `docker exec -it fos_inference_docker /bin/bash`
- **If you want to see the logs of the docker -->** `docker logs fos_inference_docker`
- **Copy the results of the docker to the host destination folder -->** `docker cp fos_inference_docker: ./output_test_files ./`
