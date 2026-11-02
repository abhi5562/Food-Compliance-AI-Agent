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
