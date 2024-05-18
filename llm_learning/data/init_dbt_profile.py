import getpass
import sys
from pathlib import Path

import click
import yaml
from jinja2 import Template
import os


@click.command()
@click.option(
    "--user",
    prompt="Snowflake username",
    default=getpass.getuser().upper().replace(".", "_"),
)
@click.option("--passwd", prompt="Snowflake password", hide_input=True)
def render_profile_yaml(
    user: str,
    passwd: str,
):
    dbt_profile_path = Path.home() / ".dbt/profiles.yml"
    if dbt_profile_path.is_file():
        with open(dbt_profile_path, "r") as f:
            existing_profile = yaml.safe_load(f)
        if "dbt_raw_data_for_llm_learning" in existing_profile.keys():
            error_msg = f"""DBT profile for dbt_raw_data_for_llm_learning
            is already initialized.  If you want to re-initialize, please delete the project
            profile then run this script again."""
            print(error_msg)
            sys.exit(2)
    else:
        # Create the file
        # Make the .dbt directory if necessary
        dbt_profile_path.parents[0].mkdir(parents=True, exist_ok=True)
        with open(dbt_profile_path, "w") as f:
            pass
        os.chmod(dbt_profile_path, 0o600)

    template_path = Path(__file__).resolve().parents[0] / "dbt_profile_template.yaml"
    with open(template_path, "r") as f:
        template = Template(f.read())

    db_type = click.prompt("Database type", default="snowflake")
    if db_type == "snowflake":
        role = click.prompt("Snowflake role", default=user + "_DEFAULT_ROLE")
        schema = click.prompt(
            "Snowflake schema for output:",
            default=f"{user}_DBT",
        )
        warehouse = click.prompt("Snowflake warehouse:", default="COMPUTE_WH")

        output = template.render(
            SNOWFLAKE_USER=user,
            USER_PASSWORD=passwd,
            ROLE=role,
            SCHEMA=schema,
            TYPE=db_type,
            WAREHOUSE=warehouse,
        )

        click.echo(f"Appending project profile to {dbt_profile_path}...")
        # Make the .dbt directory if necessary
        dbt_profile_path.parents[0].mkdir(parents=True, exist_ok=True)

        with open(dbt_profile_path, "a") as f:
            f.write("\n" + output)


if __name__ == "__main__":
    render_profile_yaml()
