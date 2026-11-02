from uuid import UUID

import psycopg
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from openai import OpenAIError
from pinecone.exceptions import PineconeException

from agents.corrective_actions import prepare_corrective_action
from agents.workflows import create_inspection_workflow
from backend.db import get_finding, get_unresolved_actions, save_corrective_proposal, save_inspection
from backend.storage import upload_image
from schemas.schema import InspectionCategory


app = FastAPI(title="Food Compliance API")
MAX_IMAGE_BYTES = 10 * 1024 * 1024


@app.post("/api/inspections", status_code=201)
def create_inspection(
    category: InspectionCategory = Form(...),
    image: UploadFile = File(...),
    description=Form(None),
):
    if image.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=415, detail="Use a JPEG, PNG, or WebP image.")

    # Validations for image size
    image_bytes = image.file.read(MAX_IMAGE_BYTES + 1)
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The image is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="The image must be at most 10 MiB.")

    try:
        reference = upload_image(image_bytes, image.content_type)
    except (BotoCoreError, ClientError) as error:
        raise HTTPException(status_code=502, detail="Could not upload the image to S3.") from error

    try:
        result = create_inspection_workflow().invoke({
            "image_bytes": image_bytes,
            "category": category,
            "mime_type": image.content_type,
        })
    except (OpenAIError, PineconeException) as error:
        raise HTTPException(
            status_code=502,
            detail="Inspection analysis failed. The photo remains stored in S3.",
        ) from error

    try:
        inspection_id = save_inspection(
            category=category.value,
            image_reference=reference,
            analysis=result["analysis"],
            retrieved_evidence=result["retrieved_evidence"],
            description=(description.strip() or None) if description is not None else None,
        )
    except psycopg.Error as error:
        raise HTTPException(
            status_code=503,
            detail="The inspection result could not be saved. The photo remains stored in S3.",
        ) from error

    return {
        "inspection_id": inspection_id,
        "category": category.value,
        "analysis": result["analysis"],
        "retrieved_evidence": result["retrieved_evidence"],
        "image_reference": reference,
    }


@app.post("/api/proposals/{finding_id}")
def create_corrective_proposal(finding_id: UUID):
    try:
        finding = get_finding(finding_id)
    except psycopg.Error as error:
        raise HTTPException(status_code=503, detail="Could not read the finding.") from error

    if finding is None:
        raise HTTPException(status_code=404, detail="Finding not found.")

    try:
        unresolved_actions = get_unresolved_actions(finding["category"])
    except psycopg.Error as error:
        raise HTTPException(status_code=503, detail="Could not read unresolved tasks.") from error

    for action in unresolved_actions:
        if action["finding_id"] == finding["id"]:
            return {
                "decision": "POSSIBLE_DUPLICATE",
                "title": None,
                "description": None,
                "existing_task_id": str(action["id"]),
                "reason": "An unresolved task already exists for this finding.",
                "finding_id": str(finding["id"]),
                "category": finding["category"],
                "task": action,
            }

    try:
        proposal = prepare_corrective_action(finding, finding["category"], unresolved_actions)
    except (OpenAIError, ValueError) as error:
        raise HTTPException(status_code=502, detail="Could not prepare a corrective proposal.") from error

    if proposal["decision"] == "POSSIBLE_DUPLICATE":
        for item in unresolved_actions:
            if str(item["id"]) == proposal["existing_task_id"]:
                action = item
                break
    else:
        try:
            action = save_corrective_proposal(finding, proposal)
        except psycopg.Error as error:
            raise HTTPException(status_code=503, detail="Could not save the corrective proposal.") from error

    return {**proposal, "task": action}
