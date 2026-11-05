import httpx

from a2a.client import ClientConfig, create_client
from a2a.helpers.proto_helpers import new_data_message
from a2a.types import Role, SendMessageRequest
from fastapi.encoders import jsonable_encoder
from google.protobuf.json_format import MessageToDict

from schemas.schema import CorrectiveProposal


async def request_corrective_proposal(
    finding, category, existing_tasks
) -> dict:
    # Convert PostgreSQL UUIDs and timestamps to JSON-compatible strings for A2A.
    payload = jsonable_encoder({
        "finding": finding,
        "category": category,
        "existing_tasks": existing_tasks,
    })

    async with httpx.AsyncClient(timeout=180) as http_client:
        client = await create_client(
            "http://127.0.0.1:8001",
            client_config=ClientConfig(
                httpx_client=http_client,
                streaming=False,
            ),
        )
        request = SendMessageRequest(
            message=new_data_message(
                payload,
                role=Role.ROLE_USER,
                media_type="application/json",
            )
        )

        async for event in client.send_message(request):
            proposal = CorrectiveProposal.model_validate(
                MessageToDict(event.message.parts[0].data)
            ).model_dump()

            return {
                **proposal,
                "finding_id": str(finding["id"]),
                "category": category,
            }

    raise ValueError("The agent returned no proposal.")
