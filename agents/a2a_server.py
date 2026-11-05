from asyncio import to_thread

from a2a.helpers.proto_helpers import new_data_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    InvalidParamsError,
    UnsupportedOperationError,
)
from google.protobuf.json_format import MessageToDict
from starlette.applications import Starlette

from agents.corrective_actions import prepare_corrective_action


proposal_skill = AgentSkill(
    id="propose_corrective_action",
    name="Prepare corrective action",
    description=(
        "Draft a correction grounded in supplied SOP evidence "
        "or identify a possible duplicate among unresolved tasks."
    ),
    tags=["corrective-actions", "restaurant-hygiene"],
    examples=["Prepare a corrective proposal for a dirty pan finding."],
    input_modes=["application/json"],
    output_modes=["application/json"],
)

agent_card = AgentCard(
    name="Corrective Action Agent",
    description="Prepares corrective proposals for supervisor review.",
    version="0.1.0",
    supported_interfaces=[
        AgentInterface(
            url="http://127.0.0.1:8001",
            protocol_binding="JSONRPC",
            protocol_version="1.0",
        )
    ],
    capabilities=AgentCapabilities(
        streaming=False,
        push_notifications=False,
    ),
    default_input_modes=["application/json"],
    default_output_modes=["application/json"],
    skills=[proposal_skill],
)


class CorrectiveActionExecutor(AgentExecutor):
    async def execute(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        message = context.message
        if (
            message is None
            or len(message.parts) != 1
            or not message.parts[0].HasField("data")
        ):
            raise InvalidParamsError(
                message="Send one structured finding request."
            )

        payload = MessageToDict(message.parts[0].data)
        required = {"finding", "category", "existing_tasks"}
        if not isinstance(payload, dict) or not required <= payload.keys():
            raise InvalidParamsError(
                message="finding, category, and existing_tasks are required."
            )

        proposal = await to_thread(
            prepare_corrective_action,
            payload["finding"],
            payload["category"],
            payload["existing_tasks"],
        )
        await event_queue.enqueue_event(
            new_data_message(
                proposal,
                media_type="application/json",
                context_id=context.context_id,
            )
        )

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        raise UnsupportedOperationError(
            message="Cancellation is not supported."
        )


request_handler = DefaultRequestHandler(
    agent_executor=CorrectiveActionExecutor(),
    task_store=InMemoryTaskStore(),
    agent_card=agent_card,
)

app = Starlette(routes=[
    *create_agent_card_routes(agent_card),
    *create_jsonrpc_routes(request_handler, rpc_url="/"),
])
