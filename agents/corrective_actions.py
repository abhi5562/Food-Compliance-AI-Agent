import json

from crewai import Agent, Crew, Task
from dotenv import load_dotenv

from agents.prompts import CORRECTIVE_ACTION_TASK_PROMPT
from schemas.schema import CorrectiveProposal


load_dotenv()


def prepare_corrective_action(finding, category, existing_tasks) -> dict:
    """Draft a corrective proposal or identify a possible existing task."""
    unresolved_tasks = [
        task for task in existing_tasks
        if task["category"] == category
        and task["status"] in ("PENDING_APPROVAL", "OPEN", "PENDING_VERIFICATION")
    ]
    agent = Agent(
        role="Corrective Action Coordinator",
        goal="Propose SOP-grounded corrections and identify possible duplicate tasks.",
        backstory="You help one restaurant's category supervisors review corrective actions.",
        llm="openai/gpt-4o-mini",
    )
    task = Task(
        description=CORRECTIVE_ACTION_TASK_PROMPT,
        expected_output=(
            "A JSON object with decision (NEW_PROPOSAL or POSSIBLE_DUPLICATE), "
            "title, description, existing_task_id, and reason. No Markdown."
        ),
        agent=agent,
        output_pydantic=CorrectiveProposal,
    )
    result = Crew(agents=[agent], tasks=[task]).kickoff(inputs={
        "category": category,
        "finding": json.dumps(finding, default=str),
        "existing_tasks": json.dumps(unresolved_tasks, default=str),
    })
    if result.pydantic is None:
        raise ValueError("The agent did not return a valid corrective proposal.")
    proposal = result.pydantic.model_dump()
    if (
        proposal["decision"] == "POSSIBLE_DUPLICATE"
        and proposal["existing_task_id"] not in {str(item["id"]) for item in unresolved_tasks}
    ):
        raise ValueError("The agent referenced a task that was not supplied.")

    return {**proposal, "finding_id": str(finding["id"]), "category": category}
