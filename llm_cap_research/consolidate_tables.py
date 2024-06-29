import glob
import pathlib

import pandas as pd
import yaml
from typer import Typer
from typing_extensions import Annotated

app = Typer()

manual_url_map = {
    "Weston-MA.pdf": "https://secureservercdn.net/50.62.195.83/env.320.myftpupload.com/wp-content/uploads/2021/08/Weston_CAP_Final_LowRes_202105261602078520.pdf",
    "Dedham-MA.pdf": "https://www.dedham-ma.gov/home/showpublisheddocument/13758/637320572656730000",
    "Melrose-MA.pdf": "https://www.cityofmelrose.org/sites/g/files/vyhlif3451/f/uploads/melrose_net_zero_action_plan_v4.pdf",
}


def get_filename_to_url_map(pdf_dir: str) -> dict:
    """
    Function to get a dictionary mapping the filename of a PDF file to its URL
    """
    # Get list of *.pdf files in data/raw
    pdf_files = glob.glob(pdf_dir + "/*.pdf")
    url_of = manual_url_map
    for pdf_file in pdf_files:
        with open(pdf_file + ".dvc", "r") as file:
            dvc_metadata = yaml.safe_load(file)

        try:
            p = pathlib.Path(pdf_file).name
            url_of[p] = dvc_metadata["deps"][0]["path"]
        except KeyError:
            if p not in url_of.keys():
                print(dvc_metadata)
                raise

    return url_of


@app.command()
def consolidate_tables(
    directory: Annotated[
        str,
        "Directory containing the *-goals.csv and *-actions.csv files to consolidate",
    ],
    pdf_dir: Annotated[str, "Directory containing the PDF files"],
):
    url_of = get_filename_to_url_map(pdf_dir)

    files = glob.glob(f"{directory}/*-goals.csv")
    goals = pd.concat([pd.read_csv(file) for file in files])
    goals["document_url"] = goals["document_url"].apply(
        lambda x: url_of[pathlib.Path(x).name]
    )
    goals.to_csv("data/processed/consolidated_goals.csv", index=False)

    # Get all the files in the directory
    files = glob.glob(f"{directory}/*-actions.csv")
    # Concatenate all the actions files
    actions = pd.concat([pd.read_csv(file) for file in files])
    actions["document_url"] = actions["document_url"].apply(
        lambda x: url_of[pathlib.Path(x).name]
    )
    # Save the consolidated actions file
    actions.to_csv("data/processed/consolidated_actions.csv", index=False)


if __name__ == "__main__":
    app()
