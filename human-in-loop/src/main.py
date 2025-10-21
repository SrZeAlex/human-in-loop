import os
from typing import TypedDict, Literal
import json
from datetime import datetime
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["GOOGLE_API_KEY"] = "Your-Google-API-Key-Here"

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)  # Fast for content screening

class ModerationState(TypedDict):
    content: str
    content_type: Literal["text", "image", "video"]
    auto_decision: Literal["approve", "flag", "reject"] | None
    risk_score: float  # 0-1, higher = more risky
    issues_found: list[str]
    human_decision: Literal["approve", "reject"] | None
    human_notes: str
    final_status: Literal["approved", "rejected", "pending", "appealed"] | None
    appeal_requested: bool | None
    appeal_decision: Literal["uphold_rejection", "overturn_rejection"] | None
    appeal_notes: str | None
    review_level: Literal["tier1", "tier2", "senior"] | None
    reviewer_id: str | None
    content_id: str | None

def auto_moderate(state: ModerationState) -> dict:
    """Automatically screen content for policy violations."""
    content = state["content"]

    prompt = f"""You are a content moderation AI. Analyze this content for policy violations:

    Content: {content}

    Check for:
    - Hate speech or harassment
    - Violence or graphic content
    - Spam or misleading information
    - Adult content
    - Copyright violations

    Respond in this format:
    DECISION: approve/flag/reject
    RISK_SCORE: 0.0-1.0
    ISSUES: comma-separated list of issues (or "none")

    Use "approve" for clearly safe content (risk < 0.3)
    Use "flag" for borderline content needing human review (risk 0.3-0.7)
    Use "reject" for clear violations (risk > 0.7)
    """

    response = llm.invoke(prompt)
    content_lines = response.content.strip().split('\n')

    # Parse response
    decision = "flag"  # Default to human review
    risk_score = 0.5
    issues = []

    for line in content_lines:
        if line.startswith("DECISION:"):
            decision = line.split(":", 1)[1].strip().lower()
        elif line.startswith("RISK_SCORE:"):
            try:
                risk_score = float(line.split(":", 1)[1].strip())
            except:
                risk_score = 0.5
        elif line.startswith("ISSUES:"):
            issues_str = line.split(":", 1)[1].strip()
            if issues_str.lower() != "none":
                issues = [i.strip() for i in issues_str.split(',')]

    return {
        "content_id": str(hash(content)), # Create a simple ID
        "auto_decision": decision,
        "risk_score": risk_score,
        "issues_found": issues
    }

def route_for_review(state: ModerationState) -> str:
    """Determine if human review is needed."""
    decision = state.get("auto_decision", "flag")
    risk_score = state.get("risk_score", 0.5)

    if decision == "approve":
        return "auto_approve"
    elif decision == "reject":
        return "auto_reject"
    else:  # Flagged for human review, route by risk
        if risk_score > 0.8:
            return "senior_review"
        elif risk_score > 0.5:
            return "tier2_review"
        return "tier1_review"

def auto_approve_node(state: ModerationState) -> dict:
    """Automatically approve safe content."""
    return {
        "final_status": "approved",
        "human_decision": None,
        "human_notes": "Auto-approved: No policy violations detected"
    }

def auto_reject_node(state: ModerationState) -> dict:
    """Automatically reject clear violations."""
    issues = ", ".join(state["issues_found"])
    return {
        "final_status": "rejected",
        "human_decision": None,
        "human_notes": f"Auto-rejected: {issues}"
    }

def tier1_review_node(state: ModerationState) -> dict:
    """Checkpoint for Tier 1 human review."""
    issues = ", ".join(state["issues_found"]) if state["issues_found"] else "General review needed"
    return {
        "final_status": "pending",
        "review_level": "tier1",
        "human_notes": f"Awaiting Tier 1 review. Issues flagged: {issues}"
    }

def tier2_review_node(state: ModerationState) -> dict:
    """Checkpoint for Tier 2 human review."""
    issues = ", ".join(state["issues_found"]) if state["issues_found"] else "General review needed"
    return {
        "final_status": "pending",
        "review_level": "tier2",
        "human_notes": f"Awaiting Tier 2 review (escalated). Issues flagged: {issues}"
    }

def senior_review_node(state: ModerationState) -> dict:
    """Checkpoint for Senior Moderator review (high-risk)."""
    issues = ", ".join(state["issues_found"]) if state["issues_found"] else "General review needed"
    return {"final_status": "pending", "review_level": "senior",
            "human_notes": f"Awaiting Senior review (high-risk). Issues flagged: {issues}"}

def apply_human_decision(state: ModerationState) -> dict:
    """Apply the human moderator's decision."""
    human_decision = state.get("human_decision")

    if human_decision == "approve":
        return {"final_status": "approved"}
    else:
        return {"final_status": "rejected"}

def appeal_review_node(state: ModerationState) -> dict:
    """Checkpoint for an appeal review - workflow pauses here."""
    original_rejection_notes = state.get("human_notes", "No notes on original rejection.")
    return {
        "final_status": "appealed",
        "appeal_notes": f"Awaiting appeal review. Original rejection reason: {original_rejection_notes}"
    }

def apply_appeal_decision(state: ModerationState) -> dict:
    """Apply the appeal reviewer's decision."""
    appeal_decision = state.get("appeal_decision")
    if appeal_decision == "overturn_rejection":
        return {"final_status": "approved"}
    else:  # uphold_rejection
        return {"final_status": "rejected"}

def check_appeal(state: ModerationState) -> str:
    """Check if an appeal has been requested for a rejected item."""
    return "appeal_review" if state.get("appeal_requested") else "log_decision"

def log_decision(state: ModerationState) -> dict:
    """Log the final decision to an audit file."""
    # Ensure final_status is set for logging purposes
    if not state.get("final_status"):
        # This can happen if a path ends without explicitly setting it
        # e.g. a rejected item that is not appealed.
        state["final_status"] = "rejected"

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "content_id": state.get("content_id"),
        "final_status": state.get("final_status"),
        "auto_decision": state.get("auto_decision"),
        "risk_score": state.get("risk_score"),
        "human_decision": state.get("human_decision"),
        "reviewer_id": state.get("reviewer_id"),
    }
    with open("moderation_audit.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    return {} # No state change needed

def create_moderation_workflow():
    # Create checkpointer for state persistence
    checkpointer = MemorySaver()

    workflow = StateGraph(ModerationState)

    # Add nodes
    workflow.add_node("auto_moderate", auto_moderate)
    workflow.add_node("auto_approve", auto_approve_node)
    workflow.add_node("auto_reject", auto_reject_node)
    workflow.add_node("apply_decision", apply_human_decision)
    workflow.add_node("tier1_review", tier1_review_node)
    workflow.add_node("tier2_review", tier2_review_node)
    workflow.add_node("senior_review", senior_review_node)
    workflow.add_node("appeal_review", appeal_review_node)
    workflow.add_node("apply_appeal", apply_appeal_decision)
    workflow.add_node("log_decision", log_decision)

    # Start with auto-moderation
    workflow.add_edge(START, "auto_moderate")

    # Conditional routing after auto-moderation
    workflow.add_conditional_edges(
        "auto_moderate",
        route_for_review,
        {
            "auto_approve": "auto_approve",
            "auto_reject": "auto_reject",
            "tier1_review": "tier1_review",
            "tier2_review": "tier2_review",
            "senior_review": "senior_review",
        }
    )

    # Auto-approved content goes to logging
    workflow.add_edge("auto_approve", "log_decision")

    # Rejected content can be appealed
    workflow.add_conditional_edges("auto_reject", check_appeal, {"appeal_review": "appeal_review", "log_decision": "log_decision"})
    workflow.add_conditional_edges("apply_decision", check_appeal, {"appeal_review": "appeal_review", "log_decision": "log_decision"})

    # Human review waits for input, then applies decision
    workflow.add_edge("tier1_review", "apply_decision")
    workflow.add_edge("tier2_review", "apply_decision")
    workflow.add_edge("senior_review", "apply_decision")

    # Appeal process
    workflow.add_edge("appeal_review", "apply_appeal")
    workflow.add_edge("apply_appeal", "log_decision")

    # Logging is the final step
    workflow.add_edge("log_decision", END)

    # Compile with checkpointer
    app = workflow.compile(checkpointer=checkpointer)
    return app, checkpointer

def get_pending_reviews(checkpointer) -> list[dict]:
    """Get all items awaiting human review from the checkpointer."""
    pending = []
    # For MemorySaver, the storage is a dict where keys are thread_ids
    for thread_id in checkpointer.storage.keys():
        config = {"configurable": {"thread_id": thread_id}}
        # `get()` retrieves the latest state for a given configuration
        state = checkpointer.get(config)
        # The actual state dict is in the 'channel_values' attribute
        state_values = state["channel_values"] if state else {}
        if state_values.get("final_status") in ("pending", "appealed"):
            status = state_values["final_status"]
            
            # Determine which notes to show based on status
            if status == "pending":
                notes = state_values.get("human_notes")
            else: # appealed
                notes = state_values.get("appeal_notes")

            pending.append({
                "thread_id": thread_id,
                "content": state_values["content"],
                "status": status,
                "review_level": state_values.get("review_level"),
                "notes": notes
            })
    return pending

def process_batch_reviews(app, reviews: list[dict]):
    """Process a batch of pending reviews with simulated human decisions."""
    print(f"--- Processing Batch of {len(reviews)} Reviews ---")
    for review in reviews:
        config = {"configurable": {"thread_id": review["thread_id"]}}
        print(f"\nReviewing Thread ID: {review['thread_id']} ({review['status']})")
        print(f"Content: '{review['content']}'")
        
        # Simulate decision based on status
        if review["status"] == "pending":
            # Simple logic: approve borderline, reject high-risk
            decision = "approve" if review["review_level"] != "senior" else "reject"
            print(f"Decision: {decision.upper()}")
            app.invoke({"human_decision": decision, "reviewer_id": "batch-proc-001"}, config)
        elif review["status"] == "appealed":
            # Simple logic: uphold all appeals in this batch
            decision = "uphold_rejection"
            print(f"Appeal Decision: UPHOLD REJECTION")
            app.invoke({"appeal_decision": decision}, config)

    print("\n--- Batch Processing Complete ---")


app, checkpointer = create_moderation_workflow()

# Test Case 1: Auto-approve (safe content)
print("=== Test 1: Safe Content ===")
config1 = {"configurable": {"thread_id": "test-1"}}

result1 = app.invoke({
    "content": "Check out this amazing sunset photo I took!",
    "content_type": "text"
}, config=config1)

print(f"Auto Decision: {result1['auto_decision']}")
print(f"Risk Score: {result1['risk_score']:.2f}")
print(f"Final Status: {result1['final_status']}\n")

# Test Case 2: Auto-reject (clear violation)
print("=== Test 2: Clear Violation ===")
config2 = {"configurable": {"thread_id": "test-2"}}

result2 = app.invoke({
    "content": "Spam message: Buy cheap products now! Click here!!!",
    "content_type": "text"
}, config=config2)

print(f"Auto Decision: {result2['auto_decision']}")
print(f"Risk Score: {result2['risk_score']:.2f}")
print(f"Final Status: {result2['final_status']}\n")

# Test Case 3: Needs human review (borderline)
print("=== Test 3: Borderline Content (Needs Human Review) ===")
config3 = {"configurable": {"thread_id": "test-3"}}

# First invocation - workflow pauses at human_review
app.invoke({
    "content": "This political statement might be controversial...",
    "content_type": "text"
}, config=config3)

# Get the state of the paused workflow
result3 = app.get_state(config3).values

print(f"Auto Decision: {result3.get('auto_decision')}")
print(f"Risk Score: {result3.get('risk_score', 0.0):.2f}")
print(f"Review Level: {result3.get('review_level')}")
print(f"Status: {result3.get('final_status')}")
print(f"Notes: {result3.get('human_notes')}\n")

print(">>> Workflow paused for human review <<<\n")

# Simulate human moderator review
print("Human moderator reviews content and decides to APPROVE\n")

# Resume workflow with human decision
result3_final = app.invoke({
    "human_decision": "approve",
    "human_notes": "Human review: Content is acceptable, approved.",
    "reviewer_id": "mod-123"
}, config=config3)

print(f"Final Status: {result3_final['final_status']}")
print(f"Human Notes: {result3_final['human_notes']}")

# Test Case 4: Rejection with Appeal
print("\n=== Test 4: Rejection with Appeal ===")
config4 = {"configurable": {"thread_id": "test-4"}}

# First invocation - auto-rejected
result4 = app.invoke({
    "content": "Get rich quick with my new crypto scheme!",
    "content_type": "text",
    "appeal_requested": True # User immediately appeals
}, config=config4)

print(f"Auto Decision: {result4['auto_decision']}")
print(f"Status: {result4['final_status']}")
print(f"Notes: {result4['appeal_notes']}\n")

print(">>> Workflow paused for appeal review <<<\n")

# Simulate appeal reviewer overturning the rejection
print("Appeal reviewer decides to OVERTURN the rejection\n")
result4_final = app.invoke({
    "appeal_decision": "overturn_rejection",
    "appeal_notes": "Appeal: Overturned. Content is satirical."
}, config=config4)

print(f"Final Status: {result4_final['final_status']}")
print(f"Appeal Notes: {result4_final['appeal_notes']}")

# Test Case 5: Escalation to Senior Moderator
print("\n=== Test 5: High-Risk Escalation ===")
config5 = {"configurable": {"thread_id": "test-5"}}

# First invocation - workflow pauses at senior_review
app.invoke({
    "content": "This content is extremely borderline and discusses sensitive topics in a way that could be seen as inciting violence.",
    "content_type": "text"
}, config=config5)

# Get the state of the paused workflow
result5 = app.get_state(config5).values

print(f"Auto Decision: {result5.get('auto_decision')}")
print(f"Risk Score: {result5.get('risk_score', 0.0):.2f}")
print(f"Review Level: {result5.get('review_level')}")
print(f"Status: {result5.get('final_status')}")
print(f"Notes: {result5.get('human_notes')}\n")

print(">>> Workflow paused for SENIOR review <<<\n")
# The workflow would now wait for a senior moderator to provide a `human_decision`.
# For brevity, we will not run the final step here, but it would be identical
# to the second step of Test Case 3.

# Test Case 6: Batch Processing
print("\n=== Test 6: Batch Processing Pending Reviews ===")

# At this point, we have two items pending from previous tests:
# - test-3 (borderline content, pending tier1 review)
# - test-5 (high-risk content, pending senior review)

# Get all pending reviews
pending_reviews = get_pending_reviews(checkpointer)
print(f"Found {len(pending_reviews)} items awaiting review.")
for item in pending_reviews:
    print(f"  - ID: {item['thread_id']}, Status: {item['status']}, Level: {item['review_level']}")

print("\n")

# Process the batch
process_batch_reviews(app, pending_reviews)
