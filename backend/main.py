from mimetypes import guess_type
from uuid import UUID
from typing import Literal

import psycopg
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from openai import OpenAIError
from pinecone.exceptions import PineconeException

from agents.corrective_actions import prepare_corrective_action
from agents.workflows import create_inspection_workflow, create_resolution_verification_workflow
from backend.db import (
    decide_proposal,
    decide_resolution,
    get_correction_image_reference,
    get_finding,
    get_open_actions,
    get_pending_verifications,
    get_resolution_context,
    get_task_image_reference,
    get_unresolved_actions,
    save_corrective_proposal,
    save_inspection,
    save_resolution_attempt,
)
from backend.storage import read_image, upload_image
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


@app.get("/api/proposals")
def list_pending_proposals(category: InspectionCategory):
    try:
        actions = get_unresolved_actions(category.value)
    except psycopg.Error as error:
        raise HTTPException(status_code=503, detail="Could not read pending proposals.") from error

    return [action for action in actions if action["status"] == "PENDING_APPROVAL"]


@app.post("/api/proposals/{task_id}/decision")
def submit_proposal_decision(
    task_id: UUID,
    category: InspectionCategory = Form(...),
    decision: Literal["APPROVE", "REJECT"] = Form(...),
    reason=Form(None),
):
    try:
        return decide_proposal(
            task_id,
            category.value,
            decision,
            (reason.strip() or None) if reason is not None else None,
        )
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except psycopg.Error as error:
        raise HTTPException(status_code=503, detail="Could not save the proposal decision.") from error


@app.get("/api/proposals/{task_id}/image")
def get_proposal_image(task_id: UUID):
    try:
        reference = get_task_image_reference(task_id)
    except psycopg.Error as error:
        raise HTTPException(status_code=503, detail="Could not read the photo reference.") from error

    if reference is None:
        raise HTTPException(status_code=404, detail="Task not found.")

    try:
        image_bytes = read_image(reference["bucket"], reference["key"])
    except (BotoCoreError, ClientError) as error:
        raise HTTPException(status_code=502, detail="Could not retrieve the photo from S3.") from error

    return Response(
        content=image_bytes,
        media_type=guess_type(reference["key"])[0] or "application/octet-stream",
    )


@app.get("/api/corrective-actions")
def list_open_actions():
    try:
        return get_open_actions()
    except psycopg.Error as error:
        raise HTTPException(
            status_code=503,
            detail="Could not read open corrective actions.",
        ) from error


@app.post("/api/corrective-actions/{task_id}/evidence", status_code=201)
def submit_correction_evidence(
    task_id: UUID,
    image: UploadFile = File(...),
    description=Form(None),
):
    try:
        context = get_resolution_context(task_id)
    except psycopg.Error as error:
        raise HTTPException(
            status_code=503,
            detail="Could not read the corrective task.",
        ) from error

    if context is None:
        raise HTTPException(status_code=404, detail="Task not found.")
    if context["status"] != "OPEN":
        raise HTTPException(
            status_code=409,
            detail="Correction evidence requires an OPEN task.",
        )

    image_bytes = image.file.read(MAX_IMAGE_BYTES + 1)
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="The image must be at most 10 MiB.",
        )

    try:
        original_image_bytes = read_image(
            context["original_s3_bucket"],
            context["original_s3_key"],
        )
        reference = upload_image(
            image_bytes,
            image.content_type,
            prefix=f"corrections/{task_id}",
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except (BotoCoreError, ClientError) as error:
        raise HTTPException(
            status_code=502,
            detail="Could not retrieve the original or store the correction photo.",
        ) from error

    description = (
        description.strip() or None
    ) if description is not None else None

    try:
        result = create_resolution_verification_workflow().invoke({
            "task_context": context,
            "original_image_bytes": original_image_bytes,
            "original_mime_type": guess_type(context["original_s3_key"])[0],
            "correction_image_bytes": image_bytes,
            "correction_mime_type": image.content_type,
            "description": description,
        })
    except OpenAIError as error:
        raise HTTPException(
            status_code=502,
            detail="Correction verification failed. The photo remains stored in S3.",
        ) from error

    try:
        attempt = save_resolution_attempt(
            task_id,
            reference,
            result["verification"],
            description,
        )
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except psycopg.Error as error:
        raise HTTPException(
            status_code=503,
            detail="Could not save correction evidence. The photo remains in S3.",
        ) from error

    return {
        "task_id": str(task_id),
        "status": "PENDING_VERIFICATION",
        "evidence": attempt,
    }


@app.get("/api/verifications")
def list_pending_verifications(category: InspectionCategory):
    try:
        return get_pending_verifications(category.value)
    except psycopg.Error as error:
        raise HTTPException(
            status_code=503,
            detail="Could not read pending verifications.",
        ) from error


@app.get("/api/corrections/{evidence_id}/image")
def get_correction_image(evidence_id: UUID):
    try:
        reference = get_correction_image_reference(evidence_id)
    except psycopg.Error as error:
        raise HTTPException(
            status_code=503,
            detail="Could not read the correction photo reference.",
        ) from error

    if reference is None:
        raise HTTPException(status_code=404, detail="Correction attempt not found.")

    try:
        image_bytes = read_image(reference["bucket"], reference["key"])
    except (BotoCoreError, ClientError) as error:
        raise HTTPException(
            status_code=502,
            detail="Could not retrieve the correction photo from S3.",
        ) from error

    return Response(
        content=image_bytes,
        media_type=guess_type(reference["key"])[0] or "application/octet-stream",
    )


@app.post("/api/verifications/{task_id}/decision")
def submit_resolution_decision(
    task_id: UUID,
    category: InspectionCategory = Form(...),
    evidence_id: UUID = Form(...),
    decision: Literal["CLOSE", "REOPEN"] = Form(...),
    reason=Form(None),
):
    try:
        return decide_resolution(
            task_id,
            category.value,
            evidence_id,
            decision,
            (reason.strip() or None) if reason is not None else None,
        )
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except psycopg.Error as error:
        raise HTTPException(
            status_code=503,
            detail="Could not save the verification decision.",
        ) from error
