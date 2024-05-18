# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from llm_learning.logging_config import configure_logging
from sqlalchemy.engine.base import Connection

logger = logging.getLogger("make_dataset")


@click.command()
@click.argument("table_name")
@click.option("--chunk_size", default=100000)
@click.option(
    "--test",
    is_flag=True,
    show_default=True,
    default=False,
    help="Output a test file with <chunk_size> rows",
)
@click.option("--profile", default=Path.home() / ".dbt/profiles.yml")
@click.option("--target", default="dev")
def main(table_name, chunk_size, test, profile, target):
    """
    Extracts all data from a table or view in the database to a local Parquet file
    """

    configure_logging(log_file=f"pipeline_logs/{table_name}.log")

    logger.info("Downloading data from database")
    target_dir = Path(__file__).resolve().parents[2] / "data" / "raw"

    # TODO - setup database connection
    conn = None

    dbt_config = dbt_profile_settings(profile, target)

    table_to_parquet(
        table_name, dbt_config, target_dir, conn, test, chunk_size=chunk_size
    )


def dbt_profile_settings(
    profiles_path: str = Path.home() / ".dbt/profiles.yml", target="dev"
) -> dict:
    with open(profiles_path, "r") as f:
        profiles = yaml.safe_load(f)

    dbt_config = profiles["{dbt_raw_data_for_llm_learning"]["outputs"][target]
    return dbt_config


def table_to_parquet(
    table_name: str,
    dbt_config: dict,
    target_dir: Path,
    conn: Connection,
    test: bool,
    chunk_size=10000,
):
    # TODO - alternate setup using dotenv if use_dbt is false
    db = dbt_config["database"]
    schema = dbt_config["schema"]

    logger.info(f"Fetching data from {db}.{schema}.{table_name}")
    if test:
        output_file = target_dir / f"{table_name}_test.parquet"
    else:
        output_file = target_dir / f"{table_name}.parquet"

    logger.info(f"Writing output to {output_file}...")
    pq_writer = None
    pq_schema = None
    num_cols = None
    num_rows = None
    query = f"SELECT * FROM {db}.{schema}.{table_name}"
    if test:
        query = query + f" limit {chunk_size}"

    for i, df in enumerate(pd.read_sql(query, conn, chunksize=chunk_size)):
        # Write chunks to multiple parquet files in a folder structure compatible with hive.

        logger.info(f"Processing chunk {i}")
        if i == 0:
            table = pa.Table.from_pandas(df)
            pq_schema = table.schema
            num_cols = df.shape[1]
            num_rows = df.shape[0]
            pq_writer = pq.ParquetWriter(output_file, pq_schema, compression="snappy")
        else:
            table = pa.Table.from_pandas(df, schema=pq_schema)
            num_rows += df.shape[0]

        pq_writer.write_table(table)

    logger.info(f"Stored {num_rows} rows and {num_cols} columns in {output_file}")


if __name__ == "__main__":
    main()
