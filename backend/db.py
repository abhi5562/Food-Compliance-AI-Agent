import os
from uuid import uuid4

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


load_dotenv()


def save_inspection(
    category: str,
    image_reference: dict,
    analysis: dict,
    retrieved_evidence: list[dict],
    description: str | None = None,
) -> str:
    """Save an inspection and its findings together, returning the inspection ID."""
    inspection_id = uuid4()
    evidence = {item["reference_id"]: item for item in retrieved_evidence}

    with psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=10) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO inspections (
                    id, category, s3_bucket, s3_key, description,
                    assessment, summary, evidence_limitations
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    inspection_id,
                    category,
                    image_reference["bucket"],
                    image_reference["key"],
                    description,
                    analysis["assessment"],
                    analysis["summary"],
                    analysis["evidence_limitations"],
                ),
            )

            for finding in analysis["findings"]:
                cursor.execute(
                    """
                    INSERT INTO findings (
                        id, inspection_id, observation, sop_requirement,
                        sop_reference, advisory_correction
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        uuid4(),
                        inspection_id,
                        finding["observation"],
                        finding["sop_requirement"],
                        Jsonb(evidence[finding["citation_reference"]]),
                        finding["advisory_correction"],
                    ),
                )

    return str(inspection_id)


def get_finding(finding_id):
    """Read one saved finding with its inspection category."""
    with psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=10) as connection:
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT f.id, f.inspection_id, f.observation, f.sop_requirement,
                       f.sop_reference, f.advisory_correction, i.category
                FROM findings f
                JOIN inspections i ON i.id = f.inspection_id
                WHERE f.id = %s
                """,
                (finding_id,),
            )
            return cursor.fetchone()


def get_unresolved_actions(category):
    """Read unresolved category tasks with the evidence behind each finding."""
    with psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=10) as connection:
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT a.id, a.finding_id, a.category, a.title, a.description,
                       a.status, a.created_at, a.closed_at,
                       f.observation, f.sop_requirement, f.sop_reference
                FROM corrective_actions a
                JOIN findings f ON f.id = a.finding_id
                WHERE a.category = %s
                  AND a.status IN ('PENDING_APPROVAL', 'OPEN', 'PENDING_VERIFICATION')
                ORDER BY a.created_at DESC
                """,
                (category,),
            )
            return cursor.fetchall()


def save_corrective_proposal(finding, proposal):
    """Save a new proposal pending supervisor approval."""
    with psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=10) as connection:
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                INSERT INTO corrective_actions (
                    id, finding_id, category, title, description, status
                ) VALUES (%s, %s, %s, %s, %s, 'PENDING_APPROVAL')
                RETURNING *
                """,
                (
                    uuid4(),
                    finding["id"],
                    finding["category"],
                    proposal["title"],
                    proposal["description"],
                ),
            )
            return cursor.fetchone()


def decide_proposal(task_id, category, decision, reason=None) -> dict:
    """Update a pending proposal and record its supervisor decision together."""
    statuses = {"APPROVE": "OPEN", "REJECT": "REJECTED"}
    if decision not in statuses:
        raise ValueError("Choose APPROVE or REJECT.")

    with psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=10) as connection:
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                UPDATE corrective_actions
                SET status = %s
                WHERE id = %s AND category = %s AND status = 'PENDING_APPROVAL'
                RETURNING *
                """,
                (statuses[decision], task_id, category),
            )
            task = cursor.fetchone()
            if task is None:
                raise ValueError("No pending proposal found for this task and category.")

            cursor.execute(
                """
                INSERT INTO approval_decisions (
                    id, task_id, approval_type, decision, reason
                ) VALUES (%s, %s, 'PROPOSAL', %s, %s)
                """,
                (uuid4(), task["id"], decision, reason),
            )
            return task


def get_task_image_reference(task_id) -> dict | None:
    """Read the original inspection photo reference for a corrective task."""
    with psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=10) as connection:
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT i.s3_bucket AS bucket, i.s3_key AS key
                FROM corrective_actions a
                JOIN findings f ON f.id = a.finding_id
                JOIN inspections i ON i.id = f.inspection_id
                WHERE a.id = %s
                """,
                (task_id,),
            )
            return cursor.fetchone()


def get_open_actions() -> list[dict]:
    """List approved tasks available for correction evidence."""
    with psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=10) as connection:
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT a.*, f.observation, f.sop_requirement, f.sop_reference
                FROM corrective_actions a
                JOIN findings f ON f.id = a.finding_id
                WHERE a.status = 'OPEN'
                ORDER BY a.created_at
                """
            )
            return cursor.fetchall()


def get_resolution_context(task_id) -> dict | None:
    """Load a task, original finding, SOP evidence, and original photo reference."""
    with psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=10) as connection:
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT a.*, f.inspection_id, f.observation, f.sop_requirement,
                       f.sop_reference, i.s3_bucket AS original_s3_bucket,
                       i.s3_key AS original_s3_key
                FROM corrective_actions a
                JOIN findings f ON f.id = a.finding_id
                JOIN inspections i ON i.id = f.inspection_id
                WHERE a.id = %s
                """,
                (task_id,),
            )
            return cursor.fetchone()


def save_resolution_attempt(task_id, image_reference, ai_verification, description=None) -> dict:
    """Save correction evidence and move an OPEN task to pending verification."""
    with psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=10) as connection:
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                UPDATE corrective_actions
                SET status = 'PENDING_VERIFICATION'
                WHERE id = %s AND status = 'OPEN'
                RETURNING id
                """,
                (task_id,),
            )
            if cursor.fetchone() is None:
                raise ValueError("Correction evidence can only be submitted for an OPEN task.")

            cursor.execute(
                """
                INSERT INTO resolution_evidence (
                    id, task_id, s3_bucket, s3_key, description, ai_verification
                ) VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (
                    uuid4(),
                    task_id,
                    image_reference["bucket"],
                    image_reference["key"],
                    description,
                    Jsonb(ai_verification),
                ),
            )
            return cursor.fetchone() 


def get_pending_verifications(category) -> list[dict]:
    """List pending tasks with their latest correction attempt."""
    with psycopg.connect(
        os.environ["DATABASE_URL"], connect_timeout=10
    ) as connection:
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT DISTINCT ON (a.id)
                       a.*, f.observation, f.sop_requirement, f.sop_reference,
                       r.id AS evidence_id,
                       r.s3_bucket AS correction_s3_bucket,
                       r.s3_key AS correction_s3_key,
                       r.description AS correction_description,
                       r.ai_verification,
                       r.created_at AS evidence_created_at
                FROM corrective_actions a
                JOIN findings f ON f.id = a.finding_id
                JOIN resolution_evidence r ON r.task_id = a.id
                WHERE a.category = %s
                  AND a.status = 'PENDING_VERIFICATION'
                ORDER BY a.id, r.created_at DESC
                """,
                (category,),
            )
            return cursor.fetchall()


def decide_resolution(
    task_id, category, evidence_id, decision, reason=None
) -> dict:
    statuses = {"CLOSE": "CLOSED", "REOPEN": "OPEN"}
    if decision not in statuses:
        raise ValueError("Choose CLOSE or REOPEN.")
    if decision == "REOPEN" and (not reason or not reason.strip()):
        raise ValueError("A reason is required when requesting rework.")

    with psycopg.connect(
        os.environ["DATABASE_URL"], connect_timeout=10
    ) as connection:
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                UPDATE corrective_actions
                SET status = %s,
                    closed_at = CASE WHEN %s = 'CLOSE'
                                     THEN CURRENT_TIMESTAMP ELSE NULL END
                WHERE id = %s AND category = %s
                  AND status = 'PENDING_VERIFICATION'
                  AND %s = (
                      SELECT id FROM resolution_evidence
                      WHERE task_id = corrective_actions.id
                      ORDER BY created_at DESC LIMIT 1
                  )
                RETURNING *
                """,
                (
                    statuses[decision], decision,
                    task_id, category, evidence_id,
                ),
            )
            task = cursor.fetchone()
            if task is None:
                raise ValueError(
                    "No pending verification matches this task, "
                    "category, and latest evidence."
                )

            cursor.execute(
                """
                INSERT INTO approval_decisions (
                    id, task_id, approval_type, decision, reason
                ) VALUES (%s, %s, 'CLOSURE', %s, %s)
                """,
                (uuid4(), task["id"], decision, reason),
            )
            return task


def get_correction_image_reference(evidence_id) -> dict | None:
    """Read the photo reference for one correction attempt."""
    with psycopg.connect(
        os.environ["DATABASE_URL"], connect_timeout=10
    ) as connection:
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT s3_bucket AS bucket, s3_key AS key
                FROM resolution_evidence
                WHERE id = %s
                """,
                (evidence_id,),
            )
            return cursor.fetchone()


def get_inspection_findings(inspection_id) -> list[dict]:
    """Read saved findings and their inspection category."""
    with psycopg.connect(
        os.environ["DATABASE_URL"], connect_timeout=10
    ) as connection:
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT f.*, i.category
                FROM findings f
                JOIN inspections i ON i.id = f.inspection_id
                WHERE f.inspection_id = %s
                """,
                (inspection_id,),
            )
            return cursor.fetchall()


def get_dashboard_tasks():
    """Read corrective tasks across all categories for the dashboard."""
    with psycopg.connect(
        os.environ["DATABASE_URL"], connect_timeout=10
    ) as connection:
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT id, category, title, description, status,
                       created_at, closed_at
                FROM corrective_actions
                ORDER BY created_at DESC
                """
            )
            return cursor.fetchall()
