### 3rd party resources
import streamlit as st
from streamlit import session_state as ss
import pandas as pd
import numpy as np
import numpy.typing as npt
from sentence_transformers import SentenceTransformer, util
from factor_analyzer import FactorAnalyzer
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

### Builtin resources
from functools import wraps
from time import time
from typing import Any, Callable, Optional, List
from zipfile import ZipFile
import os


### Own resources (see docstrings)
def report_timing(f: Callable[..., Any], report_to: Optional[str] = "terminal") -> Callable[..., Any]:
    """
    Decorator prints operation run time.
    Parameters
        - f: function to be wrapped
        - report_to: location of reporting (terminal, frontend)

    """

    @wraps(f)
    def wrap(*args: Any, **kw: Any) -> Any:
        ts = time()
        result = f(*args, **kw)
        te = time()
        # TODO implement report_to parameter correctly (user decorator factory!) so that it can be define when decorating specific function
        if report_to == "terminal":
            print("function:%r took: %2.4f sec" % (f.__name__, te - ts))
        elif report_to == "frontend":
            st.write("function:%r took: %2.4f sec" % (f.__name__, te - ts))
        else:
            print("function:%r took: %2.4f sec" % (f.__name__, te - ts))
            st.write("function:%r took: %2.4f sec" % (f.__name__, te - ts))

        return result

    return wrap


@st.cache_resource
def loadup_csv_data(file_path: str, separator: str = ";") -> pd.DataFrame:
    """Reads cvs file in as pandas object."""
    return pd.read_csv(file_path, sep=separator, low_memory=False)


@st.cache_resource
def generate_empirical_dataset(raw_dataset: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """Returns usefully formated survey response data calculated at personality facet-level."""
    # Initialize empty dataset
    emprirical_dataset = pd.DataFrame()

    # Count unique facets found in provided dataset
    count_of_facets = items["key"].nunique()

    for i in range(0, count_of_facets):
        if i == 0:
            emprirical_dataset = pd.concat(
                [
                    emprirical_dataset,
                    pd.DataFrame(
                        raw_dataset.iloc[:, i : i + 10].mean(axis=1),
                        columns=[items["key"].unique()[i]],
                    ),
                ],
                axis=1,
            )
        else:
            emprirical_dataset = pd.concat(
                [
                    emprirical_dataset,
                    pd.DataFrame(
                        raw_dataset.iloc[:, i * 10 : i * 10 + 10].mean(axis=1),
                        columns=[items["key"].unique()[i]],
                    ),
                ],
                axis=1,
            )

    return emprirical_dataset


@st.cache_resource
def load_model(model_name: str):
    """Initializes connection and downloads (tensoflow object) selected model."""
    if model_name == "Dimitre/universal-sentence-encoder":
        return hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    else:
        return SentenceTransformer(model_name)


@st.cache_resource
def vectorize_items(models: list[str], items: pd.DataFrame) -> pd.DataFrame:
    """Returns new - extended dataframe by columns with item vectors."""

    # Initialize empty dictionaries to store models and embeddings
    model_dict = {}

    try:
        # Initialize an empty DataFrame to store the embeddings
        embeddings_df = pd.DataFrame()

        for mod in models:
            # Load the model using the cached loader function
            model_dict[mod] = load_model(mod)

        # Iterate over models and encode items
        for mod, model in model_dict.items():
            # Store embeddings in temporary lists to avoid redundant computations
            temp_embeddings, temp_embeddings_rev = [], []

            for item_content, item_sign in zip(items["content"], items["sign"]):
                # Encode items
                if mod == "Dimitre/universal-sentence-encoder":
                    # For Universal Sentence Encoder, use the loaded model directly
                    item_embedding = model([item_content])[0].numpy().flatten()
                else:
                    # For other models, use SentenceTransformer's encode method
                    item_embedding = model.encode([item_content])[0]

                temp_embeddings.append(item_embedding)

                # If item is negatively keyed, reverse the embeddings
                if item_sign.startswith("neg(-)"):
                    item_embedding_rev = item_embedding * -1
                else:
                    item_embedding_rev = item_embedding
                temp_embeddings_rev.append(item_embedding_rev)

            # Add embeddings to the DataFrame
            embeddings_df[f"{mod}_embeddings"] = temp_embeddings
            embeddings_df[f"{mod}_embeddings_rev"] = temp_embeddings_rev

        # Concatenate original DataFrame with the embeddings DataFrame
        result_df = pd.concat([items, embeddings_df], axis=1)

        return result_df

    except Exception as e:
        st.error(f"Failed to initialize model or encode items: {e}")
        return None


@st.cache_resource
def compute_similarity_matrix(
    models: list[str], items: pd.DataFrame, map_model_names2nicks: dict[str, str]
) -> list[dict[str, np.ndarray | str]]:
    """Computes item embedding similarity matrices for each provided LLM."""

    similarity_matrices_collection = []

    for model_name in models:
        (
            facet_embeddings_item,
            facet_embeddings_item_rev,
            facet_embeddings_sentences,
        ) = [], [], []

        for sub_facet in items["key"].unique():
            facet_data = items[items["key"] == sub_facet]
            facet_embeddings_item.append(facet_data[model_name + "_embeddings"].mean())
            facet_embeddings_item_rev.append(facet_data[model_name + "_embeddings_rev"].mean())
            facet_embeddings_sentences.append("".join(facet_data["content"]))

        # Load the model
        model = load_model(model_name)

        # Create sentence embeddings
        if model_name == "Dimitre/universal-sentence-encoder":
            sentence_embeddings = model(facet_embeddings_sentences).numpy()
        else:
            sentence_embeddings = model.encode(facet_embeddings_sentences)

        # Create cosine similarity matrices
        cosine_similarities_item = util.pytorch_cos_sim(facet_embeddings_item, facet_embeddings_item).numpy()
        cosine_similarities_item_rev = util.pytorch_cos_sim(
            facet_embeddings_item_rev, facet_embeddings_item_rev
        ).numpy()
        cosine_similarities_sentence = util.pytorch_cos_sim(sentence_embeddings, sentence_embeddings).numpy()

        # Fill diagonal with 1
        np.fill_diagonal(cosine_similarities_item, 1)
        np.fill_diagonal(cosine_similarities_item_rev, 1)
        np.fill_diagonal(cosine_similarities_sentence, 1)

        # Store results in a dictionary
        model_results = {
            "model_name": map_model_names2nicks[model_name],
            "cosine_similarities_item": cosine_similarities_item,
            "cosine_similarities_item_rev": cosine_similarities_item_rev,
            "cosine_similarities_sentence": cosine_similarities_sentence,
        }

        similarity_matrices_collection.append(model_results)

    return similarity_matrices_collection


@st.cache_resource
def execute_hpa(
    matrix: pd.DataFrame,
    K: int = 50,
    percentile: int = 95,
    sample: Optional[int] = None,
) -> dict[str, npt.NDArray | int]:
    """Executes Horn's parallel analysis and returns hpa results within a dictionairy."""

    # Ensure that the function handles both square and rectangular data matrices appropriately
    if matrix.shape[0] == matrix.shape[1]:
        if sample is None:
            n = 1000
        else:
            n = sample
        m = matrix.shape[1]
    else:
        n, m = matrix.shape

    # Set the factor analysis parameters
    fa = FactorAnalyzer(n_factors=1, method="minres", rotation=None, use_smc=True)

    # Create arrays to store the values
    sumComponentEigens = []
    sumFactorEigens = []

    # Run the fit 'K' times over a random matrix
    for runNum in range(0, K):
        fa.fit(np.random.normal(size=(n, m)))
        sumComponentEigens.append(fa.get_eigenvalues()[0])
        sumFactorEigens.append(fa.get_eigenvalues()[1])
        # Average over the number of runs

    avgComponentEigens = np.percentile(sumComponentEigens, percentile, axis=0)
    avgFactorEigens = np.percentile(sumFactorEigens, percentile, axis=0)

    # Get the eigenvalues for the fit on supplied data
    fa.fit(matrix)
    dataEv = fa.get_eigenvalues()

    # Find the suggested stopping points
    suggestedFactors = sum((dataEv[1] - avgFactorEigens) > 0)
    suggestedComponents = sum((dataEv[0] - avgComponentEigens) > 0)

    return {
        "avgComponentEigens": avgComponentEigens,
        "dataEv": dataEv,
        "avgFactorEigens": avgFactorEigens,
        "m": m,
        "suggestedFactors": suggestedFactors,
        "suggestedComponents": suggestedComponents,
    }


@st.cache_resource
def report_hpa_results(
    hpa_results: dict[str, npt.NDArray | int],
    printEigenvalues: bool = False,
):
    """Takes and prints HPA results."""
    if printEigenvalues:
        st.write(
            "Principal component eigenvalues for random matrix:\n",
            hpa_results["avgComponentEigens"],
        )
        st.write("Factor eigenvalues for random matrix:\n", hpa_results["avgFactorEigens"])
        st.write("Principal component eigenvalues for data:\n", hpa_results["dataEv"][0])
        st.write("Factor eigenvalues for data:\n", hpa_results["dataEv"][1])

    st.write(
        "Parallel analysis suggests that the number of factors = ",
        hpa_results["suggestedFactors"],
        " and the number of components = ",
        hpa_results["suggestedComponents"],
    )


@st.cache_resource
def get_plot_object(hpa_results: dict[str, any]) -> plt.Figure:
    "Takes HPA results and return plot object(figure)."
    # Set up a figure for the scree plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the scree plot
    ax.plot([0, hpa_results["m"] + 1], [1, 1], "k--", alpha=0.3)  # Line for eigenvalue 1

    # For the random data - Components
    ax.plot(
        range(1, hpa_results["m"] + 1),
        hpa_results["avgComponentEigens"],
        "b",
        label="PC - random",
        alpha=0.4,
    )

    # For the Data - Components
    ax.scatter(range(1, hpa_results["m"] + 1), hpa_results["dataEv"][0], c="b", marker="o")
    ax.plot(range(1, hpa_results["m"] + 1), hpa_results["dataEv"][0], "b", label="PC - data")

    # For the random data - Factors
    ax.plot(
        range(1, hpa_results["m"] + 1),
        hpa_results["avgFactorEigens"],
        "g",
        label="FA - random",
        alpha=0.4,
    )

    # For the Data - Factors
    ax.scatter(range(1, hpa_results["m"] + 1), hpa_results["dataEv"][1], c="g", marker="o")
    ax.plot(range(1, hpa_results["m"] + 1), hpa_results["dataEv"][1], "g", label="FA - data")

    ax.set_xlabel("Factors/Components", fontsize=15)
    ax.set_ylabel("Eigenvalue", fontsize=15)
    # ax.set_title("Scree Plot") #NOTE not wanted here, but later on before printing PDF
    ax.legend()

    # fig.suptitle("Title of the Plot", fontsize=16) #NOTE not wanted here, but later on before printing PDF

    return fig


@st.cache_resource
def execute_and_compare_EFA_vs_PFA(
    empirical_data: pd.DataFrame,
    cosine_similarities_sentence: dict[str, np.ndarray | str],
    items: pd.DataFrame,
) -> pd.DataFrame:
    """Conducts
    1) exploratory factor analysis (based on empirical data from respondents),
    2) pseudo factor analysis (based on model embeddings from LLMs) and
    returns both results in single dataframe to contrast model fit differences.
    """
    efa_emp = FactorAnalyzer(n_factors=6, rotation="promax").fit(empirical_data)
    pfa = FactorAnalyzer(n_factors=6, rotation="promax").fit(cosine_similarities_sentence)
    results_comparison = pd.concat(
        [
            pd.DataFrame(
                np.round(efa_emp.loadings_, 3),
                index=items["key"].unique(),
            ).rename(
                columns={
                    0: "efa0",
                    1: "efa1",
                    2: "efa2",
                    3: "efa3",
                    4: "efa4",
                    5: "efa5",
                }
            ),
            pd.DataFrame(np.round(pfa.loadings_, 3), index=items["key"].unique()).rename(
                columns={
                    0: "pfa0",
                    1: "pfa1",
                    2: "pfa2",
                    3: "pfa3",
                    4: "pfa4",
                    5: "pfa5",
                }
            ),
        ],
        axis=1,
    )

    return results_comparison


def reset_analysis():
    """Callback resets given variable to its initial value (enforces user to rerun analysis)."""
    ss.similarity_matrices_collection = "init"


def initialize_new_file(
    current_dir_path: str,
    target_dir_path: str,
    target_file_name: str,
    content: Optional[Any] = None,
    extention: Optional[str] = None,
):
    """Creates a new file on the server (if provided) based on current and target path, and specified file name + extention."""
    # Construct the full path for the directory
    full_dir_path = os.path.join(current_dir_path, target_dir_path)

    # Ensure that the directory exists
    os.makedirs(full_dir_path, exist_ok=True)

    # Construct the full path for the new file
    full_file_path = os.path.join(full_dir_path, target_file_name)

    # Check if the file already exists
    if not os.path.exists(full_file_path):
        # Create a new file
        if extention == "zip":
            with ZipFile(full_file_path, "w") as new_file:
                ...
        else:
            with open(full_file_path, "w") as new_file:
                # Write content to the file if provided
                if content:
                    new_file.write(content)
            pass
    else:
        pass


def zip_files(
    current_dir_path: str,
    target_dir_path: str,
    target_file_names2zip: List[str],
    target_file_name4zip: Optional[str] = "default_name4zipped_files.zip",
):
    """Compreses source files and save zipped object on the server."""

    # Inititiaze zip object given its provided name
    zipObj = ZipFile(os.path.join(current_dir_path, target_dir_path, target_file_name4zip), "w")

    # Add multiple files to the zip
    for file_name in target_file_names2zip:
        zipObj.write(
            os.path.join(current_dir_path, target_dir_path, file_name), arcname=os.path.basename(file_name)
        )  # NOTE: arcname ensures compessed filed w/o nested in folder structure

    # Close zip file process
    zipObj.close()


def prepare_and_zip_files(
    current_dir_path: str,
    target_dir_path2dataproducts: str,
    target_file_names2zip: dict[str, str],
    target_file_name4zip: Optional[str] = "default_name4zipped_files.zip",
):
    """Prepares selected data and zips them on the server."""

    # Back up item embeddings produced by the app
    ss.hexaco_items_embedded.to_csv(
        os.path.join(current_dir_path, target_dir_path2dataproducts, target_file_names2zip["item_embeddings"]),
        sep=";",
    )

    # Define auxiliary object: labels for matrix axes
    axis_labels: pd.DataFrame = ss.hexaco_items_embedded["key"].unique()

    # Back up all similarity matrices for all models into single sheet
    with pd.ExcelWriter(
        os.path.join(
            current_dir_path,
            target_dir_path2dataproducts,
            target_file_names2zip["similarity_matrix_collection"],
        ),
        engine="auto",
    ) as writer:
        for model_results in ss.similarity_matrices_collection:
            for similarity_matrix in model_results:
                # Exluclude 'model_name' key from exporting results
                if similarity_matrix == "model_name":
                    continue
                pd.DataFrame(
                    model_results[similarity_matrix],
                    columns=axis_labels,
                    index=axis_labels,
                ).to_excel(
                    writer,
                    sheet_name=f"{similarity_matrix}_{model_results['model_name']}",
                )

    # Get list of file names to be zipped from dict
    ListOf_target_file_names2zip = [value for _, value in target_file_names2zip.items()]

    # Zip files on the server
    zip_files(
        current_dir_path,
        target_dir_path2dataproducts,
        ListOf_target_file_names2zip,
        target_file_name4zip,
    )


@st.experimental_fragment
def render_download_data_menu(**kwargs):
    """Outsources part of code to cache it as a Streamlit fragment."""  # NOTE: not sure if this actually speeds up the app

    # Initiate new zip file for all data produced (needed for context manager)
    initialize_new_file(
        kwargs["current_dir_path"],
        kwargs["target_dir_path2dataproducts"],
        kwargs["target_file_name4zip"],
        extention="zip",
    )

    # Let user download data packages available
    col1, col2 = st.columns([50, 50])

    with col1:
        # Read zip file and allow user to download all data RESOURCES
        with open(
            os.path.join(
                kwargs["current_dir_path"],
                kwargs["target_dir_path2dataresources"],
                kwargs["prezipped_data_resource_name"],
            ),
            "rb",
        ) as data2download:
            st.download_button(
                label=":floppy_disk: data resources",
                data=data2download,
                file_name=kwargs["prezipped_data_resource_name"],
                use_container_width=True,
            )

    with col2:
        # Read zip file and allow user to download all data PRODUCTS
        with open(
            os.path.join(
                kwargs["current_dir_path"],
                kwargs["target_dir_path2dataproducts"],
                kwargs["target_file_name4zip"],
            ),
            "rb",
        ) as data2download:
            st.download_button(
                # Disable button if app has not produced any data yet
                disabled=ss.similarity_matrices_collection == "init",
                label=":floppy_disk: data products",
                data=data2download,
                file_name=kwargs["target_file_name4zip"],
                help="Once available, it might take few seconds to download. **Zipping app products does not work on the server; download and run the app locally.**",
                use_container_width=True,
                # Prepare download package in the background (callback function)
                on_click=prepare_and_zip_files,
                # Pass necessary arguments to the callback function
                kwargs=dict(
                    current_dir_path=kwargs["current_dir_path"],
                    target_dir_path2dataproducts=kwargs["target_dir_path2dataproducts"],
                    target_file_name4zip=kwargs["target_file_name4zip"],
                    target_file_names2zip=kwargs["target_file_names2zip"],
                ),
            )


# Define program as main function
@report_timing
def main():
    ####### Initialization
    with st.container():
        # Initialize session state variables
        if "hexaco_items" not in ss:
            ss.hexaco_items = "init"
        if "similarity_matrices_collection" not in ss:
            ss.similarity_matrices_collection = "init"

        # Define columns formating proportions
        columns_ratio = [30, 70]

        # Define list of models (huggingface) used for vectorization
        available_models = [
            "Dimitre/universal-sentence-encoder",
            "nli-distilroberta-base-v2",
            "all-mpnet-base-v2",
            "sentence-t5-base",
            "sentence-transformers/stsb-distilroberta-base-v2",
            "microsoft/MiniLM-L12-H384-uncased",
            "uaritm/psychology_test",
        ]

        # Mapping dictionary models names2nicks
        map_model_names2nicks = {
            "Dimitre/universal-sentence-encoder": "use_dan",
            "nli-distilroberta-base-v2": "distilroberta",
            "all-mpnet-base-v2": "mpnet",
            "sentence-t5-base": "t5",
            "sentence-transformers/stsb-distilroberta-base-v2": "roberta",
            "microsoft/MiniLM-L12-H384-uncased": "miniLM",
            "uaritm/psychology_test": "psych",
        }

        # Get the directory path of the current script
        current_dir_path = os.path.dirname(__file__)

        # Define directory for the data the app uses
        target_dir_path2dataresources = "pfa_with_llms/data_resources/"

        # Define target zip file name for zipped data resources # NOTE provided on the server!
        prezipped_data_resource_name = "hexaco_data_resources.zip"

        # Define target directory for the data the app produces
        target_dir_path2dataproducts = "pfa_with_llms/data_products/"

        # Define file name for scree plots printout
        scree_plots_pdf_name = "hexaco_scree_plots.pdf"

        # Define list of expected files to be zipped to download
        target_file_names2zip = {
            "similarity_matrix_collection": "hexaco_similarity_matrix_collection.xlsx",  # .xlsx because later program uses special object to write multiple sheets into single file
            "item_embeddings": "hexaco_items_embedded.csv",
            "scree_plots": scree_plots_pdf_name,
        }

        # Define target zip file name for all data produced
        target_file_name4zip = "hexaco_data_products.zip"

        # Initialize pdf file for printed reports (if it doesn't exist yet)
        initialize_new_file(
            current_dir_path,
            target_dir_path2dataproducts,
            scree_plots_pdf_name,
        )

    ####### Info and options for users
    with st.container():
        st.title("PFA with LLMs")
        st.subheader("Pseudo-Factor Analysis with Large Language Models")
        st.markdown(
            """
            This app demostrates Python and Streamlit to present and deploy data analyses live on the cloud. 

            A part of the 'HEXACO project' was used for this purpose.
            Source datasets were slightly adjusted (semicolon separation, headers and symbols). 
            Analysis code was also slightly optimized for performance and readibility. 
            Results reported by the app might deviated from those reported in the [original project](https://osf.io/3mpzb/):
                   
            **Pseudo Factor Analysis of Language Embedding Similarity Matrices: New Ways to Model Latent Constructs**
            (by Nigel Guenole, Epifanio Damiano D'Urso, Andrew Samo, Tianjun Sun)

            """
        )
        st.code(
            """
            python==3.11.9
            +
            factor_analyzer==0.5.1
            matplotlib==3.8.4
            openpyxl==3.1.2
            numpy==1.26.4
            pandas==2.2.2
            sentence_transformers==2.7.0
            streamlit==1.33.0
            tensorflow_hub==0.16.1
            """
        )

        # Allow users to download data sources used and produced by the app
        render_download_data_menu(
            current_dir_path=current_dir_path,
            target_dir_path2dataresources=target_dir_path2dataresources,
            prezipped_data_resource_name=prezipped_data_resource_name,
            target_dir_path2dataproducts=target_dir_path2dataproducts,
            target_file_names2zip=target_file_names2zip,
            target_file_name4zip=target_file_name4zip,
        )

        st.divider()

    ####### Part 0: Prepare empirical data for analyses #######
    with st.container():
        # Initialize empty containers to format content in two columns
        col1, col2 = st.columns(columns_ratio)

        if "init" in ss.hexaco_items:
            with col1:
                if st.button("Load up source data", use_container_width=True):
                    # Load up source data files from server as pandas objects and save it to session
                    ss.hexaco_items = loadup_csv_data(
                        os.path.join(
                            current_dir_path,
                            target_dir_path2dataresources,
                            "hexaco_items.csv",
                        )
                    )

                    hexaco_raw_data = loadup_csv_data(
                        os.path.join(current_dir_path, target_dir_path2dataresources, "hexaco_raw_data.csv")
                    )

                    # Select respondents coming from US and GB only
                    hexaco_raw_data = hexaco_raw_data[hexaco_raw_data["country"].isin(["GB", "US"])]

                    # Generate empirical dataset.
                    ss.hexaco_empirical_data = generate_empirical_dataset(hexaco_raw_data, ss.hexaco_items)

                    # Rerun the script (so that program does not get stuck in this branch after clicking button)
                    st.rerun()

            with col2:
                st.info("Required datasets are stored in app's GitHub folder.")

            # Stop the program (to not run following sections)
            st.stop()

    ####### Part I: Create embeddings for PFA #######
    with st.container():
        # Let user choose which LLMs to use
        selected_models = st.multiselect(
            "Select LLM(s) to work with:",
            options=available_models,
            help="All LLMs are selected by default. Changing selection resets analysis and reporting.",
            on_change=reset_analysis,
        )
        # Otherwise program selects all LLMs
        if not selected_models:
            selected_models = available_models

        # Initialize empty containers to format content in two columns
        col1, col2 = st.columns(columns_ratio)

        if "init" in ss.similarity_matrices_collection:
            with col1:
                if st.button(
                    "Report results",
                    use_container_width=True,
                    help="Click and wait!",
                ):
                    # Vectorize items and save results into session
                    ss.hexaco_items_embedded = vectorize_items(
                        models=selected_models,
                        items=ss.hexaco_items,
                    )

                    # Compute similarity matrices and save results into session
                    ss.similarity_matrices_collection = compute_similarity_matrix(
                        selected_models, ss.hexaco_items_embedded, map_model_names2nicks
                    )

                    # Rerun the script (so that program does not get stuck in this branch after clicking button)
                    st.rerun()

            with col2:
                st.warning("First program run might take up to few minutes.")

            # Stop the program (to not run following sections)
            st.stop()

    ####### # Part II Conduct EFA and PFA #######
    with st.container():
        # Set reporting information
        plot_method_name = "Parallel Analysis Scree Plot"

        # Define tabs for each model
        model_analysis_tabs = [x for x in ss.similarity_matrices_collection[0] if "model_name" not in x] + [
            "EFA_vs_PFA_comparison"
        ]

        # Initiate plot to pdf object
        with PdfPages(os.path.join(current_dir_path, target_dir_path2dataproducts, scree_plots_pdf_name)) as pdf:
            # Evaluate empirical PA results for the item (no reverse) aggregated embeddings
            st.subheader("Parallel Analysis Scree Plots (empirical dataset)")
            # Run analysis for whole empirical dataset
            hpa_results = execute_hpa(ss.hexaco_empirical_data)
            # Print analysis results
            report_hpa_results(hpa_results)
            # Render scree plot object and render it
            plot_object = get_plot_object(hpa_results)
            st.pyplot(plot_object)
            # Add suptitle and title for printout
            plot_object.suptitle("General based on whole empirical dataset")
            plot_object.gca().set_title(plot_method_name)
            # Print to PDF object
            pdf.savefig(plot_object)

            # Run analysis, print results and render plots for each model and matrix
            for model_results in ss.similarity_matrices_collection:
                st.subheader(f"Parallel Analysis Scree Plots ({model_results['model_name']})")
                tab1, tab2, tab3, tab4 = st.tabs(model_analysis_tabs)

                with tab1:
                    temp_df = pd.DataFrame(model_results["cosine_similarities_item"])
                    hpa_results = execute_hpa(temp_df)
                    report_hpa_results(hpa_results)
                    plot_object = get_plot_object(hpa_results)
                    st.pyplot(plot_object)
                    plot_object.suptitle(f"{model_results['model_name']} > cosine_similarities_item")
                    plot_object.gca().set_title(plot_method_name)
                    pdf.savefig(plot_object)

                with tab2:
                    temp_df = pd.DataFrame(model_results["cosine_similarities_item_rev"])
                    hpa_results = execute_hpa(temp_df)
                    report_hpa_results(hpa_results)
                    plot_object = get_plot_object(hpa_results)
                    st.pyplot(plot_object)
                    plot_object.suptitle(f"{model_results['model_name']} > cosine_similarities_item_rev")
                    plot_object.gca().set_title(plot_method_name)
                    pdf.savefig(plot_object)

                with tab3:
                    temp_df = pd.DataFrame(model_results["cosine_similarities_sentence"])
                    hpa_results = execute_hpa(temp_df)
                    report_hpa_results(hpa_results)
                    plot_object = get_plot_object(hpa_results)
                    st.pyplot(plot_object)
                    plot_object.suptitle(f"{model_results['model_name']} > cosine_similarities_sentence")
                    plot_object.gca().set_title(plot_method_name)
                    pdf.savefig(plot_object)

                with tab4:
                    results_comparison = execute_and_compare_EFA_vs_PFA(
                        ss.hexaco_empirical_data,
                        model_results["cosine_similarities_sentence"],
                        ss.hexaco_items,
                    )
                    st.dataframe(results_comparison, use_container_width=True)

        st.toast("Analysis and reports complete!")


if __name__ == "__main__":
    ### Run program
    main()
    # NOTE: report for dev/test purposes
    print("---End of program---")

    # ↓↓↓ Archived resources (devs, tests, ideas)
    # def create_tabs2_IDEA(tab_names):
    #     num_tabs = len(tab_names)
    #     tabs = {}  # Create an empty dictionary to store tabs

    #     tab_stmt = f"tabs = st.tabs({tab_names})"  # Generate the statement

    #     exec(tab_stmt)

    #     for i, name in enumerate(tab_names):
    #         tabs[f"tab_{name.lower()}"] = tabs[
    #             i
    #         ]  # Store the tab in the dictionary with a dynamically generated key

    #     return tabs
