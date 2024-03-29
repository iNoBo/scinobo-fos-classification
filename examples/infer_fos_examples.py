""" 

This script imports the module fos.inference and infers some example publications.

"""
# ------------------------------------------------------------ #
import sys
sys.path.append("./src") # since it is not installed yet, we need to add the path to the module 
# -- this is for when cloning the repo
# ------------------------------------------------------------ #

from fos.pipeline.inference import create_payload, infer, process_pred
from pprint import pprint


EXAMPLE_1 = [
    {
      "id": "10.18653/v1/w19-5032",
      "title": "Embedding Biomedical Ontologies by Jointly Encoding Network Structure and Textual Node Descriptors",
      "abstract": "Network Embedding (NE) methods, which map network nodes to low-dimensional feature vectors, have wide applications in network analysis and bioinformatics. Many existing NE methods rely only on network structure, overlooking other information associated with the nodes, e.g., text describing the nodes. Recent attempts to combine the two sources of information only consider local network structure. We extend NODE2VEC, a well-known NE method that considers broader network structure, to also consider textual node descriptors using recurrent neural encoders. Our method is evaluated on link prediction in two networks derived from UMLS. Experimental results demonstrate the effectiveness of the proposed approach compared to previous work.",
      "pub_venue": "proceedings of the bionlp workshop and shared task",
      "cit_venues": ["acl",
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
            "kdd"],
      "ref_venues": ["naacl",
            "nips",
            "nucleic acids res",
            "pacific symposium on biocomputing",
            "physica a statistical mechanics and its applications",
            "proceedings of the acm conference on bioinformatics computational biology and health informatics",
            "sci china ser f",
            "the web conference"]
    }
]

EXAMPLE_2 = [
    EXAMPLE_1[0],
    {
        "id": "10.3389/frma.2023.1149834",
        "title": "SCINOBO: a novel system classifying scholarly communication in a dynamically constructed hierarchical Field-of-Science taxonomy",
        "abstract": "Classifying scientific publications according to Field-of-Science taxonomies is of crucial importance, powering a wealth of relevant applications including Search Engines, Tools for Scientific Literature, Recommendation Systems, and Science Monitoring. Furthermore, it allows funders, publishers, scholars, companies, and other stakeholders to organize scientific literature more effectively, calculate impact indicators along Science Impact pathways and identify emerging topics that can also facilitate Science, Technology, and Innovation policy-making. As a result, existing classification schemes for scientific publications underpin a large area of research evaluation with several classification schemes currently in use. However, many existing schemes are domain-specific, comprised of few levels of granularity, and require continuous manual work, making it hard to follow the rapidly evolving landscape of science as new research topics emerge. Based on our previous work of scinobo, which incorporates metadata and graph-based publication bibliometric information to assign Field-of-Science fields to scientific publications, we propose a novel hybrid approach by further employing Neural Topic Modeling and Community Detection techniques to dynamically construct a Field-of-Science taxonomy used as the backbone in automatic publication-level Field-of-Science classifiers. Our proposed Field-of-Science taxonomy is based on the OECD fields of research and development (FORD) classification, developed in the framework of the Frascati Manual containing knowledge domains in broad (first level(L1), one-digit) and narrower (second level(L2), two-digit) levels. We create a 3-level hierarchical taxonomy by manually linking Field-of-Science fields of the sciencemetrix Journal classification to the OECD/FORD level-2 fields. To facilitate a more fine-grained analysis, we extend the aforementioned Field-of-Science taxonomy to level-4 and level-5 fields by employing a pipeline of AI techniques. We evaluate the coherence and the coverage of the Field-of-Science fields for the two additional levels based on synthesis scientific publications in two case studies, in the knowledge domains of Energy and Artificial Intelligence. Our results showcase that the proposed automatically generated Field-of-Science taxonomy captures the dynamics of the two research areas encompassing the underlying structure and the emerging scientific developments.",
        "pub_venue": "Frontiers in Research Metrics and Analytics",
        "cit_venues": [
            "arxiv artificial intelligence",
            "arxiv computation and language",
            "arxiv machine learning",
            "arxiv social and information networks",
            "briefings in bioinformatics",
            "comparative and functional genomics",
            "conference of the european chapter of the association for computational linguistics",
            "cvpr"
        ],
        "ref_venues": [
            "eswc",
            "iclr",
            "icml",
            "ieee",
            "j mach learn res",
            "kdd",
            "kdd",
            "kdd"
        ]
    }
]


def infer_example_1():
    payload = create_payload(EXAMPLE_1)
    pred = infer(payload=payload)
    return process_pred(pred)


def infer_example_2():
    payload = create_payload(EXAMPLE_2)
    pred = infer(payload=payload)
    return process_pred(pred)


if __name__ == "__main__":
    pprint(infer_example_1())
    pprint(infer_example_2())