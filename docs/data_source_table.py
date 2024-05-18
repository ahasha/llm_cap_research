import logging
import warnings

import click
import pandas as pd
import yaml

warnings.filterwarnings("ignore")
logger = logging.getLogger("data_source_table")


@click.command()
@click.argument("source_yaml_path")
@click.argument("output_path")
def main(source_yaml_path: str, output_path: str):
    with open(source_yaml_path, "r") as f:
        source_dict = yaml.safe_load(f)

    table_rows = []
    for src in source_dict["sources"]:
        for table in src["tables"]:
            table_rows.append(
                dict(
                table=f"{src['database']}.{src['schema']}.{table['name']}",
                description=table.get("description", "Not provided"),
                )
            )
    
    with open(output_path, "w") as f:
        pd.DataFrame.from_dict(table_rows).to_markdown(f, index=False)


if __name__ == "__main__":
    main()