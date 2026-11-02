from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from tools.rag_tool import rag_tool
from langgraph.checkpoint.memory import InMemorySaver
import base64
from schemas.schema import ConversationState, InspectionState, InspectionResult
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy, StructuredOutputError
from pydantic import ValidationError
from agents.prompts import SOP_QUERY_SYSTEM_PROMPT, INSPECTION_SYSTEM_PROMPT

def create_rag_workflow():
    llm_with_tools = ChatOpenAI(model="gpt-4o-mini").bind_tools([rag_tool])

    tool_node = ToolNode([rag_tool])

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
    graph.add_edge(START, "chat")
    graph.add_conditional_edges("chat", tools_condition)
    graph.add_edge("tools", "chat")
    graph.add_edge("chat", END)

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
