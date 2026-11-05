from asyncio import run

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.exceptions import OutputParserException
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from tools.rag_tool import rag_tool
from langgraph.checkpoint.memory import InMemorySaver
import base64
import json
from schemas.schema import (
    ConversationState,
    SOPQueryRoute,
    InspectionState,
    InspectionResult,
    ResolutionVerificationResult,
    ResolutionVerificationState,
)
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy, StructuredOutputError
from pydantic import ValidationError
from agents.a2a_client import request_corrective_proposal
from agents.prompts import (
    SOP_QUERY_ROUTER_PROMPT,
    SOP_QUERY_SYSTEM_PROMPT,
    INSPECTION_SYSTEM_PROMPT,
    RESOLUTION_VERIFICATION_SYSTEM_PROMPT,
)

def create_rag_workflow():
    llm_with_tools = ChatOpenAI(model="gpt-4o-mini").bind_tools([rag_tool])
    router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(SOPQueryRoute)

    tool_node = ToolNode([rag_tool])

    def route_query(state: ConversationState):
        recent_messages = [
            message for message in state["messages"]
            if isinstance(message, (HumanMessage, AIMessage))
            and not (isinstance(message, AIMessage) and message.tool_calls)
        ][-6:]
        response = router_llm.invoke([
            SystemMessage(content=SOP_QUERY_ROUTER_PROMPT),
            *recent_messages,
        ])
        return "chat" if response.route == "RELATED" else "reject"

    def reject_node(state: ConversationState):
        return {"messages": [AIMessage(
            content="I can help with restaurant SOPs, hygiene, storage, and cleaning procedures."
        )]}

    def chat_node(state: ConversationState):
        messages = [
            SystemMessage(content=SOP_QUERY_SYSTEM_PROMPT),
            *state['messages']
        ]

        response = llm_with_tools.invoke(messages)

        return {'messages': [response]}

    graph = StateGraph(ConversationState)

    graph.add_node("chat", chat_node)
    graph.add_node("tools", tool_node)
    graph.add_node("reject", reject_node)
    graph.add_conditional_edges(START, route_query, {"chat": "chat", "reject": "reject"})
    graph.add_conditional_edges("chat", tools_condition)
    graph.add_edge("tools", "chat")
    graph.add_edge("reject", END)

    return graph.compile(checkpointer=InMemorySaver())

def create_inspection_workflow():
    agent = create_agent(
        model=ChatOpenAI(model="gpt-4o-mini"),
        tools=[rag_tool],
        system_prompt=INSPECTION_SYSTEM_PROMPT,
        response_format=ToolStrategy(InspectionResult, handle_errors=False),
    )

    def encode_image(state: InspectionState):
        image_base64 = base64.b64encode(state["image_bytes"]).decode()

        return {"image_b64": image_base64}

    def analyze_image(state: InspectionState):
        inputs = {
            "messages": [
                HumanMessage(content=[
                    {
                        "type": "text",
                        "text": f"Category: {state['category'].value}",
                    },
                    {
                        "type": "image",
                        "base64": state["image_b64"],
                        "mime_type": state["mime_type"],
                    },
                ])
            ]
        }
        try:
            result = agent.invoke(inputs)
        except StructuredOutputError:
            return {"analysis": None, "retrieved_evidence": []}

        evidence = []
        for message in result["messages"]:
            if (
                isinstance(message, ToolMessage)
                and message.name == "rag_tool"
                and message.status == "success"
                and message.artifact
            ):
                for passage in message.artifact:
                    evidence.append(passage)
        analysis = result.get("structured_response")
        return {
            "analysis": analysis.model_dump() if analysis is not None else None,
            "retrieved_evidence": evidence,
        }

    def validate_analysis(state: InspectionState):
        limitations = []
        try:
            analysis = InspectionResult.model_validate(state["analysis"])
        except ValidationError:
            analysis = None
            limitations.append("The model did not return a valid structured inspection result. Please retry.")

        evidence = {item["reference_id"]: item for item in state["retrieved_evidence"]}

        if not evidence:
            limitations.append("No retrieved SOP passages are available. Retry before assessing compliance.")
            
        if analysis is not None:
            for finding in analysis.findings:
                if finding.citation_reference not in evidence:
                    limitations.append("A finding references an SOP passage that was not retrieved. Please retry.")
                    break
            if analysis.assessment == "POTENTIAL_VIOLATION" and not analysis.findings:
                limitations.append("The assessment does not match the reported findings. Please retry.")
            elif analysis.assessment != "POTENTIAL_VIOLATION" and analysis.findings:
                limitations.append("The assessment does not match the reported findings. Please retry.")

        if limitations:
            if analysis is not None:
                limitations.extend(analysis.evidence_limitations)
            analysis = InspectionResult(
                assessment="INSUFFICIENT_EVIDENCE",
                summary="This inspection could not be reliably assessed.",
                findings=[],
                evidence_limitations=limitations,
            )
        return {"analysis": analysis.model_dump()}

    graph = StateGraph(InspectionState)
    graph.add_node("encode", encode_image)
    graph.add_node("analyze", analyze_image)
    graph.add_node("validate", validate_analysis)
    graph.add_edge(START, "encode")
    graph.add_edge("encode", "analyze")
    graph.add_edge("analyze", "validate")
    graph.add_edge("validate", END)

    return graph.compile()


def create_resolution_verification_workflow():
    model = ChatOpenAI(model="gpt-4o-mini").with_structured_output(
        ResolutionVerificationResult
    )

    def verify(state: ResolutionVerificationState) -> dict:
        try:
            verification = model.invoke([
                SystemMessage(content=RESOLUTION_VERIFICATION_SYSTEM_PROMPT),
                HumanMessage(content=[
                    {
                        "type": "text",
                        "text": json.dumps({
                            "task_context": state["task_context"],
                            "correction_note": state.get("description"),
                        }, default=str),
                    },
                    {
                        "type": "text",
                        "text": "Original inspection photo:",
                    },
                    {
                        "type": "image",
                        "base64": base64.b64encode(
                            state["original_image_bytes"]
                        ).decode(),
                        "mime_type": state["original_mime_type"],
                    },
                    {
                        "type": "text",
                        "text": "Correction photo:",
                    },
                    {
                        "type": "image",
                        "base64": base64.b64encode(
                            state["correction_image_bytes"]
                        ).decode(),
                        "mime_type": state["correction_mime_type"],
                    },
                ]),
            ])
        except (ValidationError, OutputParserException):
            verification = ResolutionVerificationResult(
                assessment="INSUFFICIENT_EVIDENCE",
                summary="The correction could not be reliably assessed.",
                observations=[],
                evidence_limitations=[
                    "The model returned an invalid result. Retry verification."
                ],
            )

        return {"verification": verification.model_dump()}

    graph = StateGraph(ResolutionVerificationState)
    graph.add_node("verify", verify)
    graph.add_edge(START, "verify")
    graph.add_edge("verify", END)

    return graph.compile()


def create_corrective_proposal_workflow():
    def request_proposal(state: dict) -> dict:
        proposal = run(request_corrective_proposal(
            state["finding"],
            state["category"],
            state["existing_tasks"],
        ))
        return {"proposal": proposal}

    graph = StateGraph(dict)
    graph.add_node("request_proposal", request_proposal)
    graph.add_edge(START, "request_proposal")
    graph.add_edge("request_proposal", END)

    return graph.compile()
