""" 

This script is used to create a gradio app for the SciNoBo FoS classifier. It uses the
the functions from the inference pipeline to make predictions on the input metadata.
This gradio app will be hosted on HF spaces for demo purposes.

"""

import gradio as gr

from fos.pipeline.inference import create_payload, infer, process_pred


# Define the functions to handle the inputs and outputs
def analyze_input(
    my_id: str | None,
    title: str | None,
    abstract: str | None,
    pub_venue: str | None,
    ref_venues: str | None,
    cit_venues: str | None,
    progress=gr.Progress(track_tqdm=True)
):
    if all([my_id is None, title is None, abstract is None, pub_venue is None, ref_venues is None, cit_venues is None]):
        results = {'error': 'Please provide the metadata for the publication'}
        return results
    if all([my_id == '', title == '', abstract == '', pub_venue == '', ref_venues == '', cit_venues == '']):
        results = {'error': 'Please provide the metadata for the publication'}
        return results
    # check if we can apply literal_eval to the ref_venues and cit_venues -- if not then split them at ","
    ref_venues = ref_venues.split(",")
    cit_venues = cit_venues.split(",")
    request_data = {
        'data': [{
            'id': my_id,
            'title': title,
            'abstract': abstract,
            'pub_venue': pub_venue,
            'ref_venues': ref_venues,
            'cit_venues': cit_venues
        }]
    }
    try:
        # create the payload
        payload = create_payload(request_data['data'])
        # infer the FoS
        preds = infer(payload = payload)
        # process the predictions
        response_data = process_pred(preds)
        results = {'data': response_data}
    except Exception as e:
        results = {'error': str(e)}
    return results


# Define the interface for the first tab (Text Analysis)
with gr.Blocks() as text_analysis:
    gr.Markdown("### SciNoBo Field of Science (FoS) Classification")
    id_input = gr.Textbox(label="ID (e.g. DOI)", placeholder="Enter an ID for the publication. At this demo, it is only used for reference.")
    title_input = gr.Textbox(label="Title", placeholder="Enter the title of the publication")
    abstract_input = gr.Textbox(label="Abstract", placeholder="Enter the abstract of the publication")
    pub_venue_input = gr.Textbox(label="Publication Venue", placeholder="Enter the publication venue that the publication was published")
    ref_venues_input = gr.Textbox(label="Reference Venues", placeholder="Enter the venues that the publication references, separated by commas")
    cit_veunes_input = gr.Textbox(label="Citation Venues", placeholder="Enter the venues that cite the publication, separated by commas")
    process_text_button = gr.Button("Process")
    text_output = gr.JSON(label="Output")
    process_text_button.click(analyze_input, inputs=[id_input, title_input, abstract_input, pub_venue_input, ref_venues_input, cit_veunes_input], outputs=[text_output])
    examples = gr.Examples(
        [[
            "10.18653/v1/w19-5032",
            "Embedding Biomedical Ontologies by Jointly Encoding Network Structure and Textual Node Descriptors",
            "Network Embedding (NE) methods, which map network nodes to low-dimensional feature vectors, have wide applications in network analysis and bioinformatics. Many existing NE methods rely only on network structure, overlooking other information associated with the nodes, e.g., text describing the nodes. Recent attempts to combine the two sources of information only consider local network structure. We extend NODE2VEC, a well-known NE method that considers broader network structure, to also consider textual node descriptors using recurrent neural encoders. Our method is evaluated on link prediction in two networks derived from UMLS. Experimental results demonstrate the effectiveness of the proposed approach compared to previous work.",
            "proceedings of the bionlp workshop and shared task",
            "acl,acl,aimag,arxiv artificial intelligence,arxiv computation and language,arxiv machine learning,arxiv social and information networks,briefings in bioinformatics,comparative and functional genomics,conference of the european chapter of the association for computational linguistics,cvpr,emnlp,emnlp,emnlp,emnlp,emnlp,emnlp,emnlp,eswc,iclr,icml,kdd,kdd,kdd,kdd",
            "naacl,nips,nucleic acids res,pacific symposium on biocomputing,physica a statistical mechanics and its applications,proceedings of the acm conference on bioinformatics computational biology and health informatics,sci china ser f,the web conference"
        ]],
        inputs=[id_input, title_input, abstract_input, pub_venue_input, ref_venues_input, cit_veunes_input]
    )

# Define the interface for the second tab (DOI Mode)
with gr.Blocks() as doi_mode:
    gr.Markdown("### SciNoBo Field of Science (FoS) Classification")
    doi_input = gr.Textbox(label="DOI", placeholder="Enter a valid Digital Object Identifier", interactive=False)
    gr.HTML("<span style='color:red;'>This functionality is not ready yet.</span>")

# Combine the tabs into one interface
with gr.Blocks() as demo:
    gr.TabbedInterface([text_analysis, doi_mode], ["Enter Metadata Mode", "Enter DOI Mode"])

# Launch the interface
demo.queue().launch(server_name="0.0.0.0", server_port=7860)
