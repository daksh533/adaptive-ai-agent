"""
test_rag.py

Phase 5 Checkpoint Test:
1. Verify standalone retriever on knowledge base queries.
2. Verify agent tool integration via dispatch("rag_lookup", ...).
3. Verify end-to-end LangGraph agent answering a query answerable only from the
   ingested knowledge base, citing source documents/chunks.

Run:
    python test_rag.py
"""

import sys
from rag.retriever import retrieve
from tools.registry import dispatch
from agent.loop import run_agent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def test_standalone_retriever():
    print("\n" + "=" * 60)
    print("TEST 1: Standalone RAG Retriever")
    print("=" * 60)
    query = "What are the four risk dimensions in the adaptive verification protocol?"
    results = retrieve(query, top_k=2)

    assert len(results) > 0, "No results returned by retriever!"
    print(f"Query: {query}")
    for i, r in enumerate(results, 1):
        print(f"\nMatch {i}:")
        print(f"  Source: {r['source']} (chunk {r['chunk_index']}/{r['total_chunks']})")
        print(f"  Distance: {r['distance']}")
        print(f"  Text excerpt: {r['text'][:150]}...")

    print("\n[PASSED] Standalone retriever successfully fetched relevant chunks.")


def test_tool_dispatch():
    print("\n" + "=" * 60)
    print("TEST 2: Tool Registry RAG Dispatch")
    print("=" * 60)
    tool_output = dispatch(
        "rag_lookup",
        {"query": "Who are the authors and institution of the project?", "top_k": 2}
    )
    assert "Daksh" in tool_output or "IILM" in tool_output or "project_overview" in tool_output, (
        f"Unexpected tool output: {tool_output}"
    )
    print("Tool Output Preview:")
    print(tool_output[:300] + "...")
    print("\n[PASSED] Tool dispatch for 'rag_lookup' works as expected.")


def test_agent_checkpoint():
    print("\n" + "=" * 60)
    print("TEST 3: End-to-End Agent with RAG Grounding (Phase 5 Checkpoint)")
    print("=" * 60)
    goal = (
        "Look up the project overview in the knowledge base. "
        "Find who the three authors are and what university they belong to. "
        "Cite the document source in the final answer."
    )
    state = run_agent(goal)

    print("\n" + "=" * 60)
    print("AGENT EXECUTION RESULTS")
    print("=" * 60)
    for res in state["results_so_far"]:
        print(f"\nStep: {res['step']}")
        print(f"Result:\n{res['result']}")

    # Verification: check that at least one step contains author names from knowledge base
    all_results_text = " ".join(str(r["result"]) for r in state["results_so_far"])
    assert "Daksh" in all_results_text, "Expected author 'Daksh' in agent's final answer"
    assert "IILM" in all_results_text, "Expected 'IILM' in agent's final answer"
    print("\n[PASSED] Phase 5 Checkpoint: Agent successfully answered using grounded RAG knowledge!")


if __name__ == "__main__":
    test_standalone_retriever()
    test_tool_dispatch()
    test_agent_checkpoint()

