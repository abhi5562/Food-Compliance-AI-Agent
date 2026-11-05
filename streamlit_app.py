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
            st.image(photo["image_bytes"], caption="Submitted inspection photo", width=400)

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

        if result.get("proposal_results") or result.get("proposal_errors"):
            st.subheader("Corrective tasks")

        for proposal in result.get("proposal_results", []):
            task = proposal["task"]
            if proposal["decision"] == "NEW_PROPOSAL":
                st.success(f"Awaiting supervisor approval: {task['title']}")
            else:
                st.info(f"Existing task suggested for this finding: {task['title']}")

            st.caption(f"Task: {task['id']} · Status: {task['status']}")

        for error in result.get("proposal_errors", []):
            st.warning(
                f"Inspection saved, but proposal generation failed: {error['error']}"
            )
            if error["finding_id"]:
                st.caption(f"Finding ID for manual retry: {error['finding_id']}")


def category_supervisor_page():
    st.title("Category Supervisor")
    st.caption("Approvals are unverified demo decisions; no supervisor identity is verified.")
    feedback = st.session_state.pop("supervisor_decision_feedback", None)
    if feedback:
        st.success(feedback)

    category = st.radio(
        "Category",
        list(InspectionCategory),
        format_func=lambda option: option.value,
        key="supervisor_category",
    )
    review_queue = st.radio(
        "Review queue",
        ["Pending approval", "Pending verification"],
        key="supervisor_review_queue",
    )

    if review_queue == "Pending verification":
        st.subheader("Pending verification")
        try:
            response = httpx.get(
                "http://127.0.0.1:8000/api/verifications",
                params={"category": category.value},
                timeout=30,
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            st.error(f"Could not load pending verifications: {error}")
            return

        verifications = response.json()
        if not verifications:
            st.info("No tasks awaiting verification in this category.")
            return

        verifications_by_id = {item["id"]: item for item in verifications}
        review_task_id = st.selectbox(
            "Task awaiting verification",
            list(verifications_by_id),
            format_func=lambda task_id: (
                f"{verifications_by_id[task_id]['title']} ({task_id[:8]})"
            ),
        )
        verification_task = verifications_by_id[review_task_id]
        st.subheader(verification_task["title"])
        st.write(verification_task["description"])

        st.write("**Original observation:**", verification_task["observation"])
        st.write("**SOP requirement:**", verification_task["sop_requirement"])

        reference = verification_task["sop_reference"]
        metadata = reference["metadata"]
        citation = (
            metadata.get("document_name")
            or metadata.get("source")
            or metadata.get("document_id")
        )
        if metadata.get("section") is not None:
            citation = f"{citation}, section {metadata['section']}"
        if metadata.get("page") is not None:
            citation = f"{citation}, page {int(metadata['page']) + 1}"
        if metadata.get("version") is not None:
            citation = f"{citation}, version {metadata['version']}"

        st.write("**Source:**", citation)
        with st.expander("Retrieved SOP passage"):
            st.write(reference["chunk_content"])

        st.caption(f"Correction attempt: {verification_task['evidence_id']}")
        if verification_task["correction_description"]:
            st.write("**Staff note:**", verification_task["correction_description"])

        verification = verification_task["ai_verification"]
        assessment = verification["assessment"]
        if assessment == "APPEARS_RESOLVED":
            st.success(f"AI assessment: {assessment}")
        elif assessment == "NOT_RESOLVED":
            st.error(f"AI assessment: {assessment}")
        else:
            st.warning(f"AI assessment: {assessment}")

        st.write(verification["summary"])
        for observation in verification["observations"]:
            st.write(observation)

        if verification["evidence_limitations"]:
            st.subheader("Evidence limitations")
            for limitation in verification["evidence_limitations"]:
                st.write(limitation)

        evidence_id = verification_task["evidence_id"]
        try:
            original_photo = httpx.get(
                f"http://127.0.0.1:8000/api/proposals/{review_task_id}/image",
                timeout=30,
            )
            original_photo.raise_for_status()

            correction_photo = httpx.get(
                f"http://127.0.0.1:8000/api/corrections/{evidence_id}/image",
                timeout=30,
            )
            correction_photo.raise_for_status()
        except httpx.HTTPError as error:
            st.error(f"Could not load the photos for review: {error}")
            return

        original_column, correction_column = st.columns(2)
        with original_column:
            st.image(
                original_photo.content,
                caption="Original inspection photo",
                width=400,
            )
        with correction_column:
            st.image(
                correction_photo.content,
                caption="Correction photo",
                width=400,
            )

        with st.form(f"resolution_decision_{evidence_id}"):
            reason = st.text_area("Decision reason (required for rework)")
            close_column, rework_column = st.columns(2)
            with close_column:
                close_requested = st.form_submit_button("Confirm closure")
            with rework_column:
                rework_requested = st.form_submit_button("Request rework")

        if close_requested or rework_requested:
            try:
                response = httpx.post(
                    f"http://127.0.0.1:8000/api/verifications/{review_task_id}/decision",
                    data={
                        "category": category.value,
                        "evidence_id": evidence_id,
                        "decision": "CLOSE" if close_requested else "REOPEN",
                        "reason": reason.strip(),
                    },
                    timeout=30,
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as error:
                response = error.response
                detail = response.text
                if response.headers.get("content-type", "").startswith(
                    "application/json"
                ):
                    detail = response.json().get("detail", detail)
                st.error(f"The decision could not be saved: {detail}")
                return
            except httpx.RequestError as error:
                st.error(
                    "Could not complete the decision request. "
                    f"Refresh to check the task status. Details: {error}"
                )
                return

            updated_task = response.json()
            st.session_state.supervisor_decision_feedback = (
                f"Decision saved. Task status: {updated_task['status']}."
            )
            st.rerun()
        return

    st.subheader("Pending Proposals")
    try:
        response = httpx.get(
            "http://127.0.0.1:8000/api/proposals",
            params={"category": category.value},
            timeout=30,
        )
        response.raise_for_status()
    except httpx.HTTPError as error:
        st.error(f"Could not load pending proposals: {error}")
        return

    proposals = response.json()
    if not proposals:
        st.info("No pending proposals in this category.")
        return

    proposals_by_id = {item["id"]: item for item in proposals}
    selected_id = st.selectbox(
        "Pending proposal",
        list(proposals_by_id),
        format_func=lambda task_id: f"{proposals_by_id[task_id]['title']} ({task_id[:8]})",
    )
    task = proposals_by_id[selected_id]
    st.subheader(task["title"])
    st.write(task["description"])
    st.write("**Observation:**", task["observation"])
    st.write("**SOP requirement:**", task["sop_requirement"])

    reference = task["sop_reference"]
    metadata = reference["metadata"]
    citation = metadata.get("document_name") or metadata.get("source") or metadata.get("document_id")
    if metadata.get("section") is not None:
        citation = f"{citation}, section {metadata['section']}"
    if metadata.get("page") is not None:
        citation = f"{citation}, page {int(metadata['page']) + 1}"
    if metadata.get("version") is not None:
        citation = f"{citation}, version {metadata['version']}"
    st.write("**Source:**", citation)
    with st.expander("Retrieved SOP passage"):
        st.write(reference["chunk_content"])

    try:
        photo = httpx.get(
            f"http://127.0.0.1:8000/api/proposals/{selected_id}/image",
            timeout=30,
        )
        photo.raise_for_status()
    except httpx.HTTPError as error:
        st.error(f"Could not load the original photo for review: {error}")
        return
    st.image(photo.content, caption="Original inspection photo", width=400)

    with st.form(f"proposal_decision_{selected_id}"):
        reason = st.text_area("Decision reason (optional)")
        approve_column, reject_column = st.columns(2)
        with approve_column:
            approved = st.form_submit_button("Approve")
        with reject_column:
            rejected = st.form_submit_button("Reject")

    if approved or rejected:
        try:
            response = httpx.post(
                f"http://127.0.0.1:8000/api/proposals/{selected_id}/decision",
                data={
                    "category": category.value,
                    "decision": "APPROVE" if approved else "REJECT",
                    "reason": reason.strip(),
                },
                timeout=30,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            response = error.response
            detail = response.text
            if response.headers.get("content-type", "").startswith("application/json"):
                detail = response.json().get("detail", detail)
            st.error(f"The decision could not be saved: {detail}")
            return
        except httpx.RequestError as error:
            st.error(f"Could not complete the decision request. Refresh to check the proposal status. Details: {error}")
            return

        updated_task = response.json()
        st.session_state.supervisor_decision_feedback = f"Decision saved. Task status: {updated_task['status']}."
        st.rerun()


def corrective_actions_page():
    st.title("Corrective Actions")

    result = st.session_state.pop("correction_result", None)
    if result is not None:
        evidence = result["evidence"]
        verification = evidence["ai_verification"]

        st.subheader("Last correction submission")
        st.success(f"Evidence saved. Attempt ID: {evidence['id']}")
        st.caption(f"Task: {result['task_id']}")
        st.info("Submitted for supervisor verification.")

        assessment = verification["assessment"]
        if assessment == "APPEARS_RESOLVED":
            st.success(f"AI assessment: {assessment}")
        elif assessment == "NOT_RESOLVED":
            st.error(f"AI assessment: {assessment}")
        else:
            st.warning(f"AI assessment: {assessment}")

        st.write(verification["summary"])
        for observation in verification["observations"]:
            st.write(observation)

        if verification["evidence_limitations"]:
            st.subheader("Evidence limitations")
            for limitation in verification["evidence_limitations"]:
                st.write(limitation)

    try:
        response = httpx.get(
            "http://127.0.0.1:8000/api/corrective-actions",
            timeout=30,
        )
        response.raise_for_status()
    except httpx.HTTPError as error:
        st.error(f"Could not load open tasks: {error}")
        return

    actions = response.json()
    if not actions:
        st.info("No open corrective actions.")
        return

    actions_by_id = {item["id"]: item for item in actions}
    selected_id = st.selectbox(
        "Open task",
        list(actions_by_id),
        format_func=lambda task_id: (
            f"{actions_by_id[task_id]['category']} — "
            f"{actions_by_id[task_id]['title']} ({task_id[:8]})"
        ),
    )
    task = actions_by_id[selected_id]
    st.subheader(task["title"])
    st.write("**Category:**", task["category"])
    st.write(task["description"])

    st.write("**Observation:**", task["observation"])
    st.write("**SOP requirement:**", task["sop_requirement"])

    reference = task["sop_reference"]
    metadata = reference["metadata"]
    citation = (
        metadata.get("document_name")
        or metadata.get("source")
        or metadata.get("document_id")
    )
    if metadata.get("section") is not None:
        citation = f"{citation}, section {metadata['section']}"
    if metadata.get("page") is not None:
        citation = f"{citation}, page {int(metadata['page']) + 1}"
    if metadata.get("version") is not None:
        citation = f"{citation}, version {metadata['version']}"

    st.write("**Source:**", citation)
    with st.expander("Retrieved SOP passage"):
        st.write(reference["chunk_content"])

    try:
        photo = httpx.get(
            f"http://127.0.0.1:8000/api/proposals/{selected_id}/image",
            timeout=30,
        )
        photo.raise_for_status()
    except httpx.HTTPError as error:
        st.error(f"Could not load the original photo: {error}")
        return

    st.image(photo.content, caption="Original inspection photo", width=400)

    correction_image = st.file_uploader(
        "Upload one correction photo",
        type=["jpg", "jpeg", "png", "webp"],
        key=f"correction_image_{selected_id}",
    )
    note = st.text_area(
        "Correction note (optional)",
        key=f"correction_note_{selected_id}",
    )

    if st.button("Submit correction", disabled=correction_image is None):
        with st.spinner("Verifying correction…"):
            try:
                response = httpx.post(
                    f"http://127.0.0.1:8000/api/corrective-actions/{selected_id}/evidence",
                    data={"description": note.strip()},
                    files={
                        "image": (
                            correction_image.name,
                            correction_image.getvalue(),
                            correction_image.type,
                        )
                    },
                    timeout=180,
                )
                response.raise_for_status()
            except httpx.TimeoutException:
                st.error(
                    "The request timed out. Processing may still finish; "
                    "refresh to check the task before submitting again."
                )
                return
            except httpx.RequestError as error:
                st.error(f"Could not submit correction evidence: {error}")
                return
            except httpx.HTTPStatusError as error:
                response = error.response
                detail = response.text
                if response.headers.get("content-type", "").startswith(
                    "application/json"
                ):
                    detail = response.json().get("detail", detail)
                st.error(f"Correction submission failed: {detail}")
                return

        st.session_state.correction_result = response.json()
        st.rerun()


def head_supervisor_page():
    st.title("Head Supervisor")

    try:
        response = httpx.get(
            "http://127.0.0.1:8000/api/dashboard",
            timeout=30,
        )
        response.raise_for_status()
    except httpx.HTTPError as error:
        st.error(f"Could not load dashboard tasks: {error}")
        return

    tasks = response.json()
    category = st.selectbox(
        "Category",
        ["All categories"] + [item.value for item in InspectionCategory],
    )
    if category != "All categories":
        tasks = [task for task in tasks if task["category"] == category]

    statuses = {
        "Pending approval": "PENDING_APPROVAL",
        "Open": "OPEN",
        "Pending verification": "PENDING_VERIFICATION",
        "Closed": "CLOSED",
        "Rejected": "REJECTED",
    }
    for column, (label, status) in zip(st.columns(5), statuses.items()):
        column.metric(label, sum(task["status"] == status for task in tasks))

    if not tasks:
        st.info("No corrective tasks for the selected category.")
        return

    st.dataframe(
        tasks,
        column_order=[
            "id", "category", "title", "description", "status",
            "created_at", "closed_at",
        ],
        hide_index=True,
    )


page = st.navigation([
    st.Page(conversation_page, title="Conversation", default=True),
    st.Page(image_inspection_page, title="Image Inspection"),
    st.Page(corrective_actions_page, title="Corrective Actions"),
    st.Page(category_supervisor_page, title="Category Supervisor"),
    st.Page(head_supervisor_page, title="Head Supervisor"),
])
page.run()
