""" 

This script is used to create a gradio app for the SciNoBo FoS classifier. It uses the
the functions from the inference pipeline to make predictions on the input metadata.
This gradio app will be hosted on HF spaces for demo purposes.

"""

import os
import gradio as gr
import json
import pandas as pd
import requests as req
from fos.pipeline.inference import create_payload, infer, process_pred

# retrieve HF space secrets
BACKEND_IP = os.getenv('BACKEND_IP')
BACKEND_PORT = os.getenv('BACKEND_PORT')
BACKEND_PATH = os.getenv('BACKEND_PATH')
FEEDBACK_IP = os.getenv('FEEDBACK_IP')
FEEDBACK_PORT = os.getenv('FEEDBACK_PORT')
FEEDBACK_PATH = os.getenv('FEEDBACK_PATH')
API_KEY = os.getenv('API_KEY')

# Define feedback function to send like/dislike feedback
def send_feedback(request_data, response_data, like_reaction, dislike_reaction):
    print("Sending feedback...", request_data, response_data, like_reaction, dislike_reaction)
    # Construct the feedback payload
    feedback_payload = {
        "tool_id": 1,
        "request": json.dumps(request_data),
        "result": json.dumps(response_data),
        "like": like_reaction,
        "dislike": dislike_reaction
    }
    headers = {
        'Content-Type': 'application/json',
        'x-api-key': API_KEY
    }
    try:
        # Construct feedback URL and send the POST request
        feedback_url = f"http://{FEEDBACK_IP}:{FEEDBACK_PORT}{FEEDBACK_PATH}"
        response = req.post(feedback_url, json=feedback_payload, headers=headers)
        response.raise_for_status()  # Raise an error for bad responses
        print("Feedback sent successfully.")
        return {"message": "Feedback sent successfully"}
    except req.RequestException as e:
        print("Error sending feedback:", e)
        return {"error": str(e)}

# Define feedback toggle functionality
def toggle_feedback(request_data, response_data, like_clicked, dislike_clicked):
    print("Toggling feedback...", like_clicked, dislike_clicked)

    # Determine feedback type
    like_reaction = True if like_clicked else False
    dislike_reaction = True if dislike_clicked else False

    # Send feedback to the backend
    feedback_response = send_feedback(request_data, response_data, like_reaction, dislike_reaction)

    # Return appropriate message based on the feedback response
    if 'error' in feedback_response:
        return f"Failed to send feedback: {feedback_response['error']}"
    else:
        return "Feedback sent successfully!"

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
        return results, pd.DataFrame()
    
    if all([my_id == '', title == '', abstract == '', pub_venue == '', ref_venues == '', cit_venues == '']):
        results = {'error': 'Please provide the metadata for the publication'}
        return results, pd.DataFrame()

    # Check if ref_venues and cit_venues are provided, if not set to empty list
    ref_venues = ref_venues.split(",") if ref_venues else []
    cit_venues = cit_venues.split(",") if cit_venues else []
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
        # Create the payload
        payload = create_payload(request_data['data'])
        # Infer the FoS
        preds = infer(payload=payload)
        # Process the predictions
        response_data = process_pred(preds)
        
        # Prepare results for JSON output
        json_results = {'data': response_data}
        
        # Prepare results for DataFrame output
        dataframe_results = pd.DataFrame(response_data)
        
        return json_results, dataframe_results

    except Exception as e:
        results = {'error': str(e)}
        return results, pd.DataFrame()

def analyze_input_doi(doi: str | None):
    if doi is None or doi == '':
        results = {'error': 'Please provide the DOI of the publication'}
        return results, pd.DataFrame()
    try:
        url = f"http://{BACKEND_IP}:{BACKEND_PORT}{BACKEND_PATH}{doi}"
        response = req.get(url)
        response.raise_for_status()
        doi_results = response.json()
        # Parse response to call analyze_input
        title = doi_results.get("title", "")
        abstract = doi_results.get("abstract", "")
        pub_venue = doi_results.get("pub_venue", "")
        ref_venues = ", ".join(doi_results.get("ref_venues", []))
        cit_venues = ", ".join(doi_results.get("cit_venues", []))
        metadata_results = analyze_input(doi, title, abstract, pub_venue, ref_venues, cit_venues)
        
        # Extract JSON and DataFrame outputs
        json_results, dataframe_results = metadata_results
        
        # Combine the two outputs
        combined_results = {
            "inferred_metadata": doi_results,
            "results": json_results.get("data", "")
        }

        return combined_results, dataframe_results
    except Exception as e:
        results = {'error': str(e)}
        return results, pd.DataFrame()

# Define reusable feedback and export binding function
def bind_feedback_buttons(like_button, dislike_button, json_output, feedback_message):
    like_button.click(
        toggle_feedback,
        inputs=[json_output, json_output, gr.Textbox(visible=False, value='True'), gr.Textbox(visible=False, value='False')],
        outputs=[feedback_message]
    )

    dislike_button.click(
        toggle_feedback,
        inputs=[json_output, json_output, gr.Textbox(visible=False, value='False'), gr.Textbox(visible=False, value='True')],
        outputs=[feedback_message]
    )

def bind_export_buttons(export_csv_button, export_json_button, table_output, json_output):
    export_csv_button.click(
        export_results,
        inputs=[table_output, gr.Textbox(visible=False, value='csv'), json_output],
        outputs=[gr.File()]
    )

    export_json_button.click(
        export_results,
        inputs=[table_output, gr.Textbox(visible=False, value='json'), json_output],
        outputs=[gr.File()]
    )

# export function for results
def export_results(results, export_type, original_json):
    print("Exporting results...", export_type)
    try:
        if export_type == 'csv':
            # Ensure results is a DataFrame before exporting
            try:
                if not isinstance(results, pd.DataFrame):
                    results = pd.DataFrame(results)
            except ValueError as e:
                print("Error converting results to DataFrame:", e)
                return gr.File(None), f"Error: Unable to convert results to DataFrame - {str(e)}"
            
            csv_file_path = "exported_results.csv"
            results.to_csv(csv_file_path, index=False)
            print("CSV export successful:", csv_file_path)
            return gr.File(csv_file_path)

        elif export_type == 'json':
            # Ensure original_json is serializable
            if not isinstance(original_json, (dict, list)):
                raise ValueError("Invalid data for JSON export")

            json_file_path = "exported_results.json"
            with open(json_file_path, "w") as f:
                json.dump(original_json, f, indent=4)
            print("JSON export successful:", json_file_path)
            return gr.File(json_file_path)

        else:
            print("Error: Unsupported export type or no data available.")
            return gr.File(None), "Error: Unsupported export type or no data available."
    except (IOError, ValueError) as e:
        print("Error during export:", e)
        return gr.File(None), f"Error: {str(e)}"

# CSS for styling the interface
common_css = """
.unpadded_box {
  display: none !important;
}

#like-dislike-container, #doi-like-dislike-container {
    display: flex;
    justify-content: flex-start;
    margin-top: 20px; /* Increased margin to add more space between rows */
    gap: 15px; /* Add gap between like and dislike buttons */
}

#like-btn, #dislike-btn, #like-doi-btn, #dislike-doi-btn, #export-csv-btn, #export-json-btn, 
#export-doi-csv-btn, #export-doi-json-btn, #process-btn, #process-doi-btn {
    background-color: #e0e0e0;
    font-size: 18px;
    border-radius: 8px;
    padding: 12px; /* Increased padding for better look and feel */
    margin: 10px; /* Added margin for spacing between buttons */
    max-width: 250px;
    cursor: pointer;
    border: 1px solid transparent;
    transition: background-color 0.3s, box-shadow 0.3s; /* Smooth hover transition */
}

#like-btn:hover, #dislike-btn:hover, #like-doi-btn:hover, #dislike-doi-btn:hover, 
#process-btn:hover, #process-doi-btn:hover, #export-csv-btn:hover, #export-json-btn:hover, 
#export-doi-csv-btn:hover, #export-doi-json-btn:hover {
    background-color: #d0d0d0;
    box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1); /* Add shadow on hover for depth effect */
}

.active {
    background-color: #c0c0c0;
    font-weight: bold;
    border-color: #000;
}

.feedback-message {
    font-size: 16px; /* Slightly larger for readability */
    color: #4CAF50;
    margin-top: 10px; /* Space between feedback message and buttons */
}

.gr-textbox, .gr-markdown {
    margin-top: 15px; /* Space between input elements and titles */
}

#export-container {
    margin-top: 20px; /* Add space above the export container */
    gap: 15px; /* Add gap between export buttons */
}

.output-container {
    margin-top: 30px; /* Add space above the output container */
}

.gr-row {
    margin-top: 20px; /* Spacing for each row */
}
"""

# Define the interface for the first tab (Text Analysis)
with gr.Blocks(css=common_css) as text_analysis:
    # Metadata input fields for analysis
    gr.Markdown("### SciNoBo Field of Science (FoS) Classification - Metadata Mode")
    id_input = gr.Textbox(label="ID (e.g. DOI)", placeholder="Enter an ID for the publication. At this demo, it is only used for reference.")
    title_input = gr.Textbox(label="Title", placeholder="Enter the title of the publication")
    abstract_input = gr.Textbox(label="Abstract", placeholder="Enter the abstract of the publication")
    pub_venue_input = gr.Textbox(label="Publication Venue", placeholder="Enter the publication venue that the publication was published")
    ref_venues_input = gr.Textbox(label="Reference Venues", placeholder="Enter the venues that the publication references, separated by commas")
    cit_venues_input = gr.Textbox(label="Citation Venues", placeholder="Enter the venues that cite the publication, separated by commas")
    process_text_button = gr.Button("Process", elem_id="process-btn")

    # Group related elements in a single container
    with gr.Group(visible=False, elem_id="output-container") as output_container:
        # Output fields for displaying results
        table_output = gr.DataFrame(label="Table View", elem_id="table_view", interactive=False)
        json_view_output = gr.JSON(label="JSON View", elem_id="json_view")

        # Feedback buttons container for user reaction
        reaction_label = gr.Markdown("**Reaction**")
        with gr.Row(elem_id="like-dislike-container"):
            like_button = gr.Button("👍 Like", elem_id="like-btn")
            dislike_button = gr.Button("👎 Dislike", elem_id="dislike-btn")
            feedback_message = gr.Markdown("")

        # Export options container
        export_label = gr.Markdown("**Export Options**")
        with gr.Row(elem_id="export-container"):
            export_csv_button = gr.Button("📄 Export as CSV", elem_id="export-csv-btn")
            export_json_button = gr.Button("📝 Export as JSON", elem_id="export-json-btn")

    # Bind export buttons to export function for Metadata mode
    bind_export_buttons(export_csv_button, export_json_button, table_output, json_view_output)

    # Bind process button to analyze input function
    process_text_button.click(
        analyze_input,
        inputs=[id_input, title_input, abstract_input, pub_venue_input, ref_venues_input, cit_venues_input],
        outputs=[json_view_output, table_output]  # Ensure both outputs are specified here
    ).then(
        lambda: gr.update(visible=True),  # Show entire container after the first request
        inputs=[],
        outputs=[output_container]
    )
    
    # Bind feedback buttons for Metadata Mode
    bind_feedback_buttons(like_button, dislike_button, json_view_output, feedback_message)

    # Examples for user reference
    examples = gr.Examples(
        [[
            "10.18653/v1/w19-5032",
            "Embedding Biomedical Ontologies by Jointly Encoding Network Structure and Textual Node Descriptors",
            "Network Embedding (NE) methods, which map network nodes to low-dimensional feature vectors, have wide applications in network analysis and bioinformatics. Many existing NE methods rely only on network structure, overlooking other information associated with the nodes, e.g., text describing the nodes. Recent attempts to combine the two sources of information only consider local network structure. We extend NODE2VEC, a well-known NE method that considers broader network structure, to also consider textual node descriptors using recurrent neural encoders. Our method is evaluated on link prediction in two networks derived from UMLS. Experimental results demonstrate the effectiveness of the proposed approach compared to previous work.",
            "proceedings of the bionlp workshop and shared task",
            "acl, acl, aimag, arxiv artificial intelligence, arxiv computation and language, arxiv machine learning, arxiv social and information networks, briefings in bioinformatics, comparative and functional genomics, conference of the european chapter of the association for computational linguistics, cvpr, emnlp, emnlp, emnlp, emnlp, eswc, iclr, icml, kdd, kdd, kdd, kdd",
            "naacl, nips, nucleic acids res, pacific symposium on biocomputing, physica a statistical mechanics and its applications, proceedings of the acm conference on bioinformatics computational biology and health informatics, sci china ser f, the web conference"
        ]],
        inputs=[id_input, title_input, abstract_input, pub_venue_input, ref_venues_input, cit_venues_input]
    )

# Define the interface for the second tab (DOI Mode)
with gr.Blocks(css=common_css) as doi_mode:
    gr.Markdown("### SciNoBo Field of Science (FoS) Classification - DOI Mode")
    doi_input = gr.Textbox(label="DOI", placeholder="Enter a valid Digital Object Identifier")
    doi_process_button = gr.Button("Process")

    # Group related elements in a single container
    with gr.Group(visible=False) as doi_output_container:
        table_doi_output = gr.DataFrame(label="Table View", elem_id="table_view_doi", interactive=False)
        json_doi_output = gr.JSON(label="JSON View", elem_id="json_view_doi")

        doi_reaction_label = gr.Markdown("**Reaction**")
        with gr.Row(elem_id="doi-like-dislike-container"):
            like_doi_button = gr.Button("👍 Like", elem_id="like-doi-btn")
            dislike_doi_button = gr.Button("👎 Dislike", elem_id="dislike-doi-btn")
            doi_feedback_message = gr.Markdown("")

        doi_export_label = gr.Markdown("**Export Options**")
        with gr.Row(elem_id="export-doi-container"):
            export_doi_csv_button = gr.Button("📄 Export as CSV", elem_id="export-doi-csv-btn")
            export_doi_json_button = gr.Button("📝 Export as JSON", elem_id="export-doi-json-btn")

    doi_process_button.click(
        analyze_input_doi,
        inputs=[doi_input],
        outputs=[json_doi_output, table_doi_output]
    ).then(
        lambda: gr.update(visible=True),  # Show entire container after the first request
        inputs=[],
        outputs=[doi_output_container]
    )

    # Bind feedback buttons for DOI Mode
    bind_feedback_buttons(like_doi_button, dislike_doi_button, json_doi_output, doi_feedback_message)

    # Bind export buttons to export function for DOI mode
    bind_export_buttons(export_doi_csv_button, export_doi_json_button, table_doi_output, json_doi_output)

# Combine the tabs into one interface
with gr.Blocks(css=common_css) as demo:
    gr.TabbedInterface([text_analysis, doi_mode], ["Metadata Mode", "DOI Mode"])

# Launch the interface
demo.queue().launch(server_name="0.0.0.0", server_port=7860)
