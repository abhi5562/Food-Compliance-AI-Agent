from uuid import uuid4

import streamlit as st
import httpx
from langchain_core.messages import HumanMessage

from agents.workflows import create_rag_workflow
from schemas.schema import InspectionCategory


st.set_page_config(page_title="Food Compliance Assistant", page_icon="🍽️")


def conversation_page():
    st.title("Food Compliance Assistant")

    if "messages" not in st.session_state:
        st.session_state.workflow = create_rag_workflow()
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid4())

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.workflow.invoke(
                    {"messages": [HumanMessage(content=query)]},
                    config={"configurable": {"thread_id": st.session_state.thread_id}},
                )
                answer = result["messages"][-1].content
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})


def image_inspection_page():
    def reset_inspection():
        st.session_state.inspection_category = InspectionCategory.FOOD_STORAGE
        st.session_state.inspection_upload_key = str(uuid4())
        st.session_state.inspection_note = ""
        st.session_state.pop("inspection_result", None)
        st.session_state.pop("inspection_photo", None)

    st.title("Image Inspection")
    if "inspection_upload_key" not in st.session_state:
        st.session_state.inspection_upload_key = str(uuid4())

    category = st.radio(
        "Category",
        list(InspectionCategory),
        format_func=lambda option: option.value,
        key="inspection_category",
    )
    image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"],
        key=st.session_state.inspection_upload_key,
    )
    note = st.text_area("Inspection note (optional)", key="inspection_note")
    submitted = st.button("Submit", disabled=image is None)
    st.button("Reset", on_click=reset_inspection)

    if submitted:
        st.session_state.pop("inspection_result", None)
        st.session_state.pop("inspection_photo", None)
        image_bytes = image.getvalue()
        with st.spinner("Processing inspection…"):
            try:
                response = httpx.post(
                    "http://127.0.0.1:8000/api/inspections",
                    data={"category": category.value, "description": note.strip()},
                    files={"image": (image.name, image_bytes, image.type)},
                    timeout=180,
                )
                response.raise_for_status()
            except httpx.TimeoutException:
                st.error("The request timed out. Processing may still finish; check the backend before submitting again.")
                st.stop()
            except httpx.RequestError as error:
                st.error(f"Could not complete the inspection request. Ensure FastAPI is running. Details: {error}")
                st.stop()
            except httpx.HTTPStatusError as error:
                response = error.response
                if response.headers.get("content-type", "").startswith("application/json"):
                    detail = response.json().get("detail", response.text)
                else:
                    detail = response.text
                st.error(f"Inspection failed ({response.status_code}): {detail}")
                st.stop()

        result = response.json()
        st.session_state.inspection_result = result
        st.session_state.inspection_photo = {
            **result["image_reference"],
            "image_bytes": image_bytes,
        }

    if "inspection_photo" in st.session_state:
        photo = st.session_state.inspection_photo
        if "image_bytes" in photo:
            st.image(photo["image_bytes"], caption="Submitted inspection photo")

    if "inspection_result" in st.session_state:
        result = st.session_state.inspection_result
        analysis = result["analysis"]
        evidence = {}
        for passage in result["retrieved_evidence"]:
            evidence[passage["reference_id"]] = passage

        st.subheader("Inspection Results")
        if result.get("inspection_id"):
            st.success(f"Inspection saved. ID: {result['inspection_id']}")
        st.write("**Category:**", result["category"])
        assessment = analysis["assessment"]
        if assessment == "NO_ISSUE_OBSERVED":
            st.success(f"Assessment: {assessment}")
        elif assessment == "POTENTIAL_VIOLATION":
            st.error(f"Assessment: {assessment}")
        else:
            st.warning(f"Assessment: {assessment}")
        st.write(analysis["summary"])

        for index, finding in enumerate(analysis["findings"], start=1):
            metadata = evidence[finding["citation_reference"]]["metadata"]
            citation = (
                metadata.get("document_name")
                or metadata.get("source")
                or metadata.get("document_id")
                or finding["citation_reference"]
            )
            if metadata.get("section") is not None:
                citation = f"{citation}, section {metadata['section']}"
            if metadata.get("page") is not None:
                citation = f"{citation}, page {int(metadata['page']) + 1}"
            if metadata.get("version") is not None:
                citation = f"{citation}, version {metadata['version']}"

            st.subheader(f"Finding {index}")
            st.write("**Observation:**", finding["observation"])
            st.write("**SOP requirement:**", finding["sop_requirement"])
            st.write("**Source:**", citation)
            st.write("**Advisory correction:**", finding["advisory_correction"])

        if analysis["evidence_limitations"]:
            st.subheader("Evidence Limitations")
            for limitation in analysis["evidence_limitations"]:
                st.write(limitation)


page = st.navigation([
    st.Page(conversation_page, title="Conversation", default=True),
    st.Page(image_inspection_page, title="Image Inspection"),
])
page.run()
