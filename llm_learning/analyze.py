import json
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Iterator, List, Optional

import dotenv
import pandas as pd
import tiktoken
import typer
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import ValidationError
from typing_extensions import Annotated

from dvclive import Live
from llm_learning.logging_config import configure_logging

logger = logging.getLogger(__name__)

dotenv_path = ".env"
dotenv.load_dotenv(dotenv_path)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
# NOTE: langsmith auto-tracing requires LANGCHAIN_API_KEY set in .env

app = typer.Typer()

CONTEXT_WINDOW = {
    "gpt-4o": 2 * 4096,
    "gpt-3.5-turbo": 16000,
}

# cl100k_base is appropriate for
# gpt-4, gpt-3.5-turbo, text-embedding-ada-002, text-embedding-3-small, text-embedding-3-large
# Source https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
TOKENIZER = {
    "gpt-4o": "o200k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-4": "cl100k_base",
}


class ActionCategory(str, Enum):
    Governance = "G"
    Zoning = "Z"
    Buildings = "B"
    Energy = "E"
    Transportation = "T"
    Waste = "W"
    Conservation = "C"


class Goal(BaseModel):
    """Information about a strategic planning Goal.

    Goals are quantified outcomes necessary to meet emissions targets and resilience goals.
    If the goal does not mention a quantitative target and a target year, you should skip it or classify it as an Action instead.
    """

    document_goal_id: Optional[str] = Field(
        description="If a unique identifier for the Goal ID is given in the text record it here."
    )
    action_category: ActionCategory = Field(
        description="""
        The Action Category the goal is associated with, e.g. Buildings, Transportation, Waste, Governance, Conservation, or Energy.
        Select Energy only if no other more specific emission category is mentioned, or if the goal pertains specifically to electricity.
        Governance is for goals related to municipal staffing, policies, or processes to support execution of the Climate Action Plan.
        """
    )
    year: Optional[int] = Field(
        default=None,
        description="The year by which the goal should be achieved.",
        ge=1990,
        le=2100,
    )
    description: str = Field(
        description="A summary description of the goal, which must include a quantitative target and a target year.",
    )
    context: str = Field(
        description="Verbatim text from the provided document on which the Goal description is based"
    )
    context_page: int = Field(
        description="The page number of the document that the context string was drawn from."
    )


class Action(BaseModel):
    """Information about an action item described in the document."""

    id: str = Field(
        description="If a Unique Identifier for the action is given in the text, reecord it here."
    )
    emissions_category: ActionCategory = Field(
        description="The category of emissions the strategy is associated with"
    )
    owner: Optional[List[str]] = Field(
        default=None,
        description="The entities or individuals responsible for the action, if mentioned in the text",
    )
    description: str = Field(description="A summary description of the action")
    context: str = Field(
        description="Verbatim text from the provided document on which the Action description is based"
    )
    context_page: int = Field(
        description="The page number of the document that the context string was drawn from."
    )


class Results(BaseModel):
    goals: Optional[List[Goal]] = []
    actions: Optional[List[Action]] = []


def results_to_goals_table(results, page, municipality):
    records = [
        {
            **g.dict(),
            "document_url": page.metadata["source"],
            "municipality": municipality,
        }
        for g in results.goals
    ]
    return pd.DataFrame(records)


def results_to_actions_table(results, page, municipality):
    records = [
        {
            **a.dict(),
            "document_url": page.metadata["source"],
            "municipality": municipality,
        }
        for a in results.actions
    ]
    return pd.DataFrame(records)


# Function to count tokens in a document
def count_tokens(text, tokenizer):
    tokens = tokenizer.encode(text)
    return len(tokens)


def document_iterator(pdf: Path, max_tokens: int, model: str) -> Iterator[Document]:
    # Initialize the tokenizer.
    encoding = TOKENIZER[model]
    tokenizer = tiktoken.get_encoding(encoding)

    from langchain_community.document_loaders import PyMuPDFLoader

    loader = PyMuPDFLoader(pdf)
    pages = loader.load()
    full_document_content = ""
    first_page = 0
    last_page = 0
    chunk_tokens = 0
    for page in pages:
        page_number = int(page.metadata["page"])
        new_content = f"PAGE {page_number}\n\n{page.page_content}"
        _full_doc_content = full_document_content + "\n\n" + new_content
        expanded_chunk_tokens = count_tokens(_full_doc_content, tokenizer)
        if expanded_chunk_tokens > max_tokens:
            # Adding this page would take you over the token limit, so yield the current chunk.
            document = Document(
                full_document_content,
                metadata={
                    "first_page": first_page,
                    "last_page": last_page,
                    "source": str(pdf),
                    "chunk_tokens": chunk_tokens,
                },
            )
            yield document
            # Reset the working chunk to only the current page.
            first_page = page_number
            last_page = page_number
            full_document_content = new_content
            chunk_tokens = count_tokens(new_content, tokenizer)
        else:  # chunk is still under the token limit, add onto it.
            chunk_tokens = expanded_chunk_tokens
            last_page = page_number
            full_document_content = _full_doc_content

    document = Document(
        full_document_content,
        metadata={
            "first_page": first_page,
            "last_page": last_page,
            "source": str(pdf),
            "chunk_tokens": chunk_tokens,
        },
    )
    yield document


def get_prompt_template() -> ChatPromptTemplate:
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
                1. Goals;
                2. Action items to achieve the goals; """
                "If the text does not explicitly describe a goal or action, do not make up an answer."
                "Only extract relevant information from the text."
                "However, be exhaustive, and extract all goals and actions that are explicitly mentioned in the text. "
                "There will often be multiple goals and actions mentioned on each page. "
                "If the value of an attribute you are asked to extract is not present in the text, return null for the attribute's value.",
            ),
            # Please see the how-to about improving performance with
            # reference examples.
            # MessagesPlaceholder('examples'),
            ("human", "{text}"),
        ]
    )
    return prompt


def check_output(tool_output):
    """Check for parse error or failure to call the tool"""

    # Error with parsing
    if tool_output["parsing_error"]:
        # Report back output and parsing errors
        print("Parsing error!")
        raw_output = str(tool_output["raw"].content)
        error = tool_output["parsing_error"]
        raise ValueError(
            f"Error parsing your output! Be sure to invoke the tool. Output: {raw_output}. \n Parse error: {error}"
        )

    # Tool was not invoked
    elif not tool_output["parsed"]:
        print("Failed to invoke tool!")
        raise ValueError(
            "You did not use the provided tool! Be sure to invoke the tool to structure the output."
        )
    return tool_output


def insert_errors(inputs):
    breakpoint()
    """Insert errors for tool parsing in the messages"""

    # Get errors
    error = inputs["error"]
    messages = inputs["messages"]
    messages += [
        (
            "assistant",
            f"Retry. You are required to fix the parsing errors: {error} \n\n You must invoke the provided tool.",
        )
    ]
    return {
        "messages": messages,
        "context": inputs["context"],
    }


def parse_output(solution):
    """When we add 'include_raw=True' to structured output,
    it will return a dict w 'raw', 'parsed', 'parsing_error'."""

    return solution["parsed"]


@app.command()
def extract(
    pdf: Path,
    llm_model: Annotated[str, typer.Option()] = "gpt-3.5-turbo",
    context_length: Annotated[int, typer.Option()] = 2 * 4096,
):
    if llm_model not in CONTEXT_WINDOW.keys():
        raise ValueError(f"Invalid LLM model: {llm_model}")
    if context_length > CONTEXT_WINDOW[llm_model]:
        raise ValueError(
            f"Context length {context_length} exceeds maximum for model {llm_model}: {CONTEXT_WINDOW[llm_model]}"
        )
    if not pdf.suffix == ".pdf":
        ValueError(f"File {pdf} is not a PDF file.")
    if not pdf.exists():
        ValueError(f"PDF file {pdf} does not exist.")

    goals_path = Path("data/processed", pdf.stem + "-goals.csv")
    actions_path = Path("data/processed", pdf.stem + "-actions.csv")
    json_path = Path("data/processed", pdf.stem + ".json")
    municipality = pdf.stem.replace("-", ", ")

    configure_logging(f"pipeline_logs/{pdf.stem}.log")

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=llm_model, temperature=0)
    prompt = get_prompt_template()
    # runnable_raw = (
    #     prompt
    #     | llm.with_structured_output(schema=Results, include_raw=True)
    #     | check_output
    # )
    # fallback_chain = prompt | insert_errors | runnable_raw
    # N = 3  # Max re-tries
    # runnable_with_retry = runnable_raw.with_fallbacks(
    #     fallbacks=[fallback_chain] * N, exception_key="error"
    # )

    # chain = runnable_with_retry | parse_output
    chain = prompt | llm.with_structured_output(schema=Results)

    goals_tables = []
    actions_tables = []
    json_output = []
    tokens_processed = 0
    for chunk in document_iterator(pdf, context_length, llm_model):
        logger.info(
            f"Processing pages {chunk.metadata['first_page']} to {chunk.metadata['last_page']} from {chunk.metadata['source']}"
        )
        try:
            try:
                result = chain.invoke({"text": chunk.page_content})
            except ValidationError as e:
                logger.error(f"A validation error occurred: {str(e)}")
                continue

            json_output.append(
                {
                    "source": chunk.metadata["source"],
                    "first_page": chunk.metadata["first_page"],
                    "last_page": chunk.metadata["last_page"],
                    "chunk_tokens": chunk.metadata["chunk_tokens"],
                    "content": chunk.page_content,
                    "municipality": municipality,
                    "results": result.dict(),
                }
            )
            tokens_processed += chunk.metadata["chunk_tokens"]

            goals_tables.append(results_to_goals_table(result, chunk, municipality))
            actions_tables.append(results_to_actions_table(result, chunk, municipality))
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")

    goals_df = pd.concat(goals_tables)
    actions_df = pd.concat(actions_tables)

    with Live() as live:
        muni_dir = municipality.replace(", ", "-")
        live.log_metric(f"{muni_dir}/goals_extracted", goals_df.shape[0])
        live.log_metric(f"{muni_dir}/actions_extracted", actions_df.shape[0])
        live.log_metric(f"{muni_dir}/doc_tokens_processed", tokens_processed)
    logger.info("Writing goals to %s", goals_path)
    goals_df.to_csv(goals_path, index=False)
    logger.info("Writing actions to %s", actions_path)
    actions_df.to_csv(actions_path, index=False)

    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2)


if __name__ == "__main__":
    app()
