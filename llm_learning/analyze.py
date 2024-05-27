import logging
import os
from enum import Enum
from pathlib import Path
from typing import Iterator, List, Optional

import dotenv
import pandas as pd
import typer
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import ValidationError
from typing_extensions import Annotated

from llm_learning.logging_config import configure_logging

logger = logging.getLogger(__name__)

dotenv_path = ".env"
dotenv.load_dotenv(dotenv_path)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
# NOTE: langsmith auto-tracing requires LANGCHAIN_API_KEY set in .env

app = typer.Typer()


class EmissionsCategory(str, Enum):
    Buildings = "B"
    Energy = "E"
    Transportation = "T"
    Waste = "W"
    LandUse = "L"


class Goal(BaseModel):
    """Information about a strategic planning Goal.

    Goals are broad, quantifiable outcomes necessary to meet emissions targets and resilience goals.
    If the goal does not mention a quantitative target or a target year, you should classify it as a Strategy instead.
    """

    id: str = Field(
        description="Unique Identifier.  If a Goal ID is given in the text, use this.  Otherwise, the first letter should be emissions_category letter."
    )
    emissions_category: EmissionsCategory = Field(
        description="""
        The category of emissions the goal is associated with, e.g. Buildings, Transportation, Waste or Land Use, Energy.
        Select Energy only if no other more specific emission category is mentioned, or if the goal pertains to electricity.
        """
    )
    year: Optional[int] = Field(
        default=None,
        description="The year by which the goal should be achieved.",
        ge=2024,
        le=2050,
    )
    description: str = Field(
        description="A summary description of the goal which must include a measurable, quantitative outcome",
    )
    context: str = Field(
        description="Verbatim text from the provided document on which the Goal description is based"
    )


class Strategy(BaseModel):
    """Information about a strategic planning Strategy.

    Strategies define general approaches to make progress toward goals.
    They should be specific, but need not necessarily be quantifiable.
    They are frequently explicitly labeled as a "Strategy" in the text.
    However, they are often described in the text as "Goals", but with no associated quantitative outcome mentioned.
    """

    id: str = Field(
        description="Unique Identifier.  If a Strategy ID is given in the text, use this.  Otherwise, the first letter should be emissions_category letter."
    )
    emissions_category: EmissionsCategory = Field(
        description="""
        The category of emissions the goal is associated with, e.g. Buildings, Transportation, Waste or Land Use, Energy.
        Select Energy only if no other more specific emission category is mentioned, or if the goal pertains to electricity.
        """
    )
    related_goals: Optional[List[str]] = Field(
        description="A list of goal ids that this strategy is related to"
    )
    description: str = Field(default=None, description="A description of the strategy")
    context: str = Field(
        description="Verbatim text from the provided document on which the Strategy description is based"
    )


class Action(BaseModel):
    """Information about a strategic planning Action.

    Actions are specific, time-bound steps to implement strategies.
    If the Action does not mention an owner or is very general, you should classify it as a Strategy instead.
    """

    id: str = Field(
        description="Unique Identifier.  First category should be emissions_category letter."
    )
    emissions_category: EmissionsCategory = Field(
        description="The category of emissions the strategy is associated with"
    )
    owner: Optional[List[str]] = Field(
        default=None,
        description="The organization or individuals responsible for the action, if known",
    )
    related_stragegies: Optional[List[str]] = Field(
        default=None,
        description="A list of strategy ids that this action is related to",
    )
    description: str = Field(description="A summary description of the action")
    context: str = Field(
        description="Verbatim text from the provided document on which the Action description is based"
    )


class Results(BaseModel):
    goals: Optional[List[Goal]] = []
    strategies: Optional[List[Strategy]] = []
    actions: Optional[List[Action]] = []


def results_to_goals_table(results, page, municipality):
    records = [
        {
            **g.dict(),
            "document_url": page.metadata["source"],
            "page": page.metadata["page"],
            "municipality": municipality,
        }
        for g in results.goals
    ]
    return pd.DataFrame(records)


def results_to_strategies_table(results, page, municipality):
    records = [
        {
            **s.dict(),
            "document_url": page.metadata["source"],
            "page": page.metadata["page"],
            "municipality": municipality,
        }
        for s in results.strategies
    ]
    return pd.DataFrame(records)


def results_to_actions_table(results, page, municipality):
    records = [
        {
            **a.dict(),
            "document_url": page.metadata["source"],
            "page": page.metadata["page"],
            "municipality": municipality,
        }
        for a in results.actions
    ]
    return pd.DataFrame(records)


def chunk(pdf: Path) -> Iterator[Document]:
    from langchain_community.document_loaders import PyMuPDFLoader

    loader = PyMuPDFLoader(pdf)
    pages = loader.load()
    for page in pages:
        yield page


def get_prompt():
    from langchain_core.prompts import ChatPromptTemplate

    # Define a custom prompt to provide instructions and any additional context.
    # 1) You can add examples into the prompt template to improve extraction quality
    # 2) Introduce additional parameters to take context into account (e.g., include metadata
    #    about the document from which the text was extracted.)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert municipal climate action planner "
                """Read the provided Climate Action Plan document and extract the following information described in the text:
                1. Emissions reductions Goals;
                2. Strategies to achieve the goals;
                3. Action items related to the Strategies; """
                "If the text does not describe a goal, strategy, or action, do not make up an answer."
                "Only extract relevant information from the text."
                "If the value of an attribute you are asked to extract is not present in the text, return null for the attribute's value."
                "The same text cannot be both a goal and a strategy.  Please distinguish them based on their definitions.",
            ),
            # Please see the how-to about improving performance with
            # reference examples.
            # MessagesPlaceholder('examples'),
            ("human", "{text}"),
        ]
    )
    return prompt


@app.command()
def chunks(pdf: Path):
    for page in chunk(pdf):
        print(f"## Page {page.metadata['page']} from {page.metadata['source']}")
        print(page.page_content)


@app.command()
def extract(
    pdf: Path,
    llm_model: Annotated[str, typer.Option()] = "gpt-3.5-turbo",
):
    if not pdf.suffix == ".pdf":
        ValueError(f"File {pdf} is not a PDF file.")
    if not pdf.exists():
        ValueError(f"PDF file {pdf} does not exist.")

    goals_path = Path("data/processed", pdf.stem + "-goals.csv")
    strategies_path = Path("data/processed", pdf.stem + "-strategies.csv")
    actions_path = Path("data/processed", pdf.stem + "-actions.csv")
    json_path = Path("data/processed", pdf.stem + ".json")
    municipality = pdf.stem.replace("-", ", ")

    configure_logging(f"pipeline_logs/{pdf.stem}.log")

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=llm_model, temperature=0)
    prompt = get_prompt()
    runnable = prompt | llm.with_structured_output(schema=Results)

    goals_tables = []
    strategies_tables = []
    actions_tables = []
    json_output = []
    for page in chunk(pdf):
        logger.info(
            f"Processing page {page.metadata['page']} from {page.metadata['source']}"
        )
        # Display page content to the user
        print(page.page_content)
        # Ask user whether to proceed with AI extraction
        try:
            # user_input = input("Proceed with AI extraction? (y/n/q): ")
            # if user_input.lower() == "q":
            #     break
            # elif user_input.lower() != "y":
            #     continue

            logger.info("Extracting from page %s...", page.metadata["page"])
            try:
                result = runnable.invoke({"text": page.page_content})
            except ValidationError as e:
                logger.error(f"A validation error occurred: {str(e)}")
                continue

            json_output.append(
                {
                    "source": page.metadata["source"],
                    "page": page.metadata["page"],
                    "content": page.page_content,
                    "municipality": municipality,
                    "results": result.dict(),
                }
            )
            json_output.str = json_output.dumps()
            with open(json_path, "w") as f:
                f.write(json_output.str)

            goals_tables.append(results_to_goals_table(result, page, municipality))
            strategies_tables.append(
                results_to_strategies_table(result, page, municipality)
            )
            actions_tables.append(results_to_actions_table(result, page, municipality))
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")

    goals_df = pd.concat(goals_tables)
    strategies_df = pd.concat(strategies_tables)
    actions_df = pd.concat(actions_tables)

    logger.info("Writing goals to %s", goals_path)
    goals_df.to_csv(goals_path, index=False)
    logger.info("Writing strategies to %s", strategies_path)
    strategies_df.to_csv(strategies_path, index=False)
    logger.info("Writing actions to %s", actions_path)
    actions_df.to_csv(actions_path, index=False)


if __name__ == "__main__":
    app()
