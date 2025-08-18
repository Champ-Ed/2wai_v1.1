import streamlit as st

import os
import asyncio
import uuid
from typing import TypedDict, List, Optional, Dict, Any, Callable
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
# from dotenv import load_dotenv
import openai
from llama_index.vector_stores.deeplake import DeepLakeVectorStore
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import VectorStoreQuery, MetadataFilters, ExactMatchFilter
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver
import traceback
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from typing_extensions import Annotated
import operator
import json


def _read_persona_template(path: Optional[str]) -> str:
    """Read persona prompt template from file if available; otherwise return a default.
    Supports placeholders {CONTEXT} and {SUMMARY}. If not present, we will inject under
    the lines starting with 'Context:' and 'Summary:' respectively.
    """
    try:
        p = Path(path or "calum_prompt_v2.1.yaml")
        if p.exists():
            return p.read_text(encoding="utf-8")
    except Exception:
        pass
    # Fallback default
    return (
        "You are Calum Worthy, a witty activist and actor. Answer as Calum.\n"
        "Context:\n{CONTEXT}\n\nSummary:\n{SUMMARY}\n"
    )

class DiskTidStore:
    """Very small JSON store for per-thread scratchpad persistence across sessions."""
    def __init__(self, base_dir: Optional[str] = None, debug: bool = False):
        self.base = Path(base_dir or "thread_checkpoints")
        self.base.mkdir(parents=True, exist_ok=True)
        self.debug = debug

    def _path(self, tid: str) -> Path:
        # sanitize filename (tid should already be a uuid-like string)
        fname = f"{tid}.json"
        return self.base / fname

    def load(self, tid: str) -> Dict[str, Any]:
        try:
            p = self._path(tid)
            if not p.exists():
                return {}
            data = json.loads(p.read_text(encoding="utf-8"))
            if self.debug:
                print(f"[DISK] Loaded thread {tid}: keys={list(data.keys())}")
            return data if isinstance(data, dict) else {}
        except Exception as e:
            if self.debug:
                print(f"[DISK] Load error for {tid}: {e}")
            return {}

    def save(self, tid: str, data: Dict[str, Any]):
        try:
            p = self._path(tid)
            p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            if self.debug:
                sp = data.get("scratchpad", [])
                print(f"[DISK] Saved thread {tid}: scratchpad_len={len(sp) if isinstance(sp, list) else 0}")
        except Exception as e:
            if self.debug:
                print(f"[DISK] Save error for {tid}: {e}")

# ---------- Typed states ----------
class AgentState(TypedDict, total=False):
    user_input: str
    session: Dict[str, Any]
    # Use a simple list with manual deduplication - operator.add may be causing issues
    scratchpad: List[str]
    selected_context: str
    compressed_history: str
    agent_context: str
    response: str

# ---------- Embedding helpers ----------
@lru_cache(maxsize=4096)
def cached_openai_embedding(text: str) -> tuple:
    """Synchronous wrapper (cached) for OpenAI embeddings.
       We return a tuple because lists are unhashable for caching; conversion handled by caller."""
    if not text or len(text) > 8192:
        return tuple()
    resp = openai.embeddings.create(
        model="text-embedding-3-small",
        input=[text]
    )
    return tuple(resp.data[0].embedding)

async def get_embedding(text: str) -> List[float]:
    # run the cached call in a threadpool so it doesn't block the event loop
    emb_tuple = await asyncio.to_thread(cached_openai_embedding, text)
    return list(emb_tuple)

# ---------- DeepLake store with batching + async wrappers ----------
class DeepLakePDFStore:
    def __init__(self, path: Optional[str] = None, commit_batch: int = 8, debug: bool = False):
        org_id = st.secrets.get("ACTIVELOOP_ORG_ID", "")
        path = path or f"hub://{org_id}/calum_v10"
        self.dataset_path = path
        self.commit_batch = commit_batch
        self.debug = debug or (os.getenv("DEBUG_CONVO") == "1")
        # Simple cache for recent queries
        self._query_cache = {}
        self._cache_max_size = 50
        # Per-event-loop locks to avoid cross-loop issues in Streamlit
        self._locks: Dict[int, asyncio.Lock] = {}
        if self.debug:
            print(f"[DL] Init store path={path} batch={commit_batch}")

        # Unified read/write store
        self.vector_store = DeepLakeVectorStore(
            dataset_path=path,
            token=st.secrets.get("ACTIVELOOP_TOKEN", ""),
            read_only=False
        )
        # self.vector_store = DeepLakeVectorStore(
        #     dataset_path=path,
        #     read_only=False
        # )
        self.index = VectorStoreIndex.from_vector_store(self.vector_store)

    def _loop_lock(self) -> asyncio.Lock:
        """Return an asyncio.Lock bound to the current running loop."""
        loop = asyncio.get_running_loop()
        lid = id(loop)
        lock = self._locks.get(lid)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[lid] = lock
        return lock

    def debug_nodes(self, nodes: List[TextNode]):
        if not self.debug:
            return
        print(f"[DL] inserting {len(nodes)} node(s):")
        for n in nodes:
            txt = n.get_content() if hasattr(n, "get_content") else getattr(n, "text", "")
            print(f"  - id={getattr(n, 'id_', None)} len={len(txt)} meta={getattr(n, 'metadata', {})}")

    async def add_memory(self, agent: str, text: str):
        if not text:
            return
        if self.debug:
            print(f"[DL] add_memory queued text='{text[:60]}...'")
        
        # Chunk text
        chunks = self.chunk_text(text)
        nodes = []
        for chunk in chunks:
            node = TextNode(
                id_=str(uuid.uuid4()),
                text=chunk,
                metadata={
                    "agent": agent,
                    "type": "memory",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
            nodes.append(node)
        
        # Debug nodes
        self.debug_nodes(nodes)
        
        # Insert nodes via LlamaIndex in a thread to avoid blocking
        try:
            async with self._loop_lock():
                await asyncio.to_thread(self._sync_insert_nodes, nodes)
            if self.debug:
                print("[DL] LlamaIndex insert_nodes done (async)")
        except Exception as e:
            if self.debug:
                print(f"[DL] LlamaIndex insert_nodes error: {e}")

    def _sync_insert_nodes(self, nodes):
        """Synchronous wrapper for node insertion."""
        self.index.insert_nodes(nodes)

    async def flush(self):
        # No-op: using vector store managed commits
        return

    @traceable(name="rag_query")
    async def rag_query(self, query: str, top_k: int = 5, agent_id_filter: Optional[str] = None) -> List[str]:
        if not query:
            return []
            
        # Check cache first
        cache_key = f"{query[:100]}:{top_k}:{agent_id_filter}"
        if cache_key in self._query_cache:
            if self.debug:
                print(f"[DL] rag_query cache hit for: {query[:50]}")
            return self._query_cache[cache_key]
            
        if self.debug:
            print(f"[DL] rag_query q='{query}' k={top_k} agent_filter={agent_id_filter}")

        try:
            retriever = self.index.as_retriever(similarity_top_k=top_k)

            agent_value = str(agent_id_filter if agent_id_filter is not None else "1")
            retriever.filters = MetadataFilters(filters=[ExactMatchFilter(key="agent", value=agent_value)])

            def _sync_retrieve(q):
                return retriever.retrieve(q)

            # Guard with per-loop lock
            async with self._loop_lock():
                nodes = await asyncio.wait_for(
                    asyncio.to_thread(_sync_retrieve, query),
                    timeout=0.8
                )

            results = [node.get_content() for node in nodes]
            
            # Cache the results
            if len(self._query_cache) >= self._cache_max_size:
                # Remove oldest entry
                oldest_key = next(iter(self._query_cache))
                del self._query_cache[oldest_key]
            self._query_cache[cache_key] = results

            if self.debug:
                print(f"[DL] rag_query fetched {len(nodes)} results in time")

            return results
            
        except asyncio.TimeoutError:
            if self.debug:
                print(f"[DL] rag_query timed out for query: {query[:50]}")
            return []  # Return empty results on timeout
        except Exception as e:
            # Retry once on transient DeepLake read errors likely due to recent writes
            if "Unable to read sample" in str(e) or "chunks" in str(e):
                if self.debug:
                    print("[DL] rag_query transient read error, retrying once...")
                try:
                    await asyncio.sleep(0.25)
                    retriever = self.index.as_retriever(similarity_top_k=top_k)
                    agent_value = str(agent_id_filter if agent_id_filter is not None else "1")
                    retriever.filters = MetadataFilters(filters=[ExactMatchFilter(key="agent", value=agent_value)])
                    def _sync_retrieve2(q):
                        return retriever.retrieve(q)
                    async with self._loop_lock():
                        nodes = await asyncio.to_thread(_sync_retrieve2, query)
                    results = [node.get_content() for node in nodes]
                    # Cache
                    if len(self._query_cache) >= self._cache_max_size:
                        oldest_key = next(iter(self._query_cache))
                        del self._query_cache[oldest_key]
                    self._query_cache[cache_key] = results
                    if self.debug:
                        print(f"[DL] rag_query retry fetched {len(nodes)} results")
                    return results
                except Exception as e2:
                    if self.debug:
                        print(f"[DL] rag_query retry error: {e2}")
            if self.debug:
                print(f"[DL] rag_query error: {e}")
            return []

    async def debug_raw_query(self, text: str, k: int = 5, agent="1"):
        q_emb = Settings.embed_model.get_text_embedding(text)
        raw = self.vector_store.query(
            VectorStoreQuery(
                query_embedding=q_emb,
                similarity_top_k=k,
                filters=MetadataFilters(filters=[ExactMatchFilter(key="agent", value=str(agent))])
            )
        )
        print("RAW ids:", getattr(raw, "ids", None))
        print("RAW sims:", getattr(raw, "similarities", None))
        print("RAW metas:", getattr(raw, "metadatas", None))

    def chunk_text(self, text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
        if not text:
            return []
        chunks = []
        words = text.split()
        current_chunk = []
        for word in words:
            if sum(len(w) + 1 for w in current_chunk) + len(word) + 1 <= chunk_size:
                current_chunk.append(word)
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

# ---------- Orchestrator (LangGraph) ----------
class OrchestratedConversationalSystem:
    def __init__(self, session: Dict, agent: str = "1"):
        self.session = session
        # Initialize debug early
        self.debug = session.get("debug", False) or (os.getenv("DEBUG_CONVO") == "1")
        
        # Initialize persistent checkpointer
        db_cfg = self.session.get("checkpoint_db", os.getenv("LANGGRAPH_CHECKPOINT_DB", "checkpoints.sqlite"))
        
        try:
            # Use session-based persistence for Streamlit compatibility
            from langgraph.checkpoint.memory import MemorySaver
            self.checkpointer = MemorySaver()
            self._checkpointer_cm = None
            
            # Try to restore checkpoints from session state
            if hasattr(st, 'session_state') and hasattr(st.session_state, 'checkpoints'):
                # Restore previous checkpoints
                self.checkpointer.storage = dict(st.session_state.checkpoints)  # Make a copy
                self.session["checkpoint_info"] = f"memory-saver (restored): {len(st.session_state.checkpoints)} checkpoints"
                if self.debug:
                    print(f"[CHECKPOINT] Restored {len(st.session_state.checkpoints)} checkpoints from session")
                    print(f"[CHECKPOINT] Checkpoint keys: {list(st.session_state.checkpoints.keys())}")
            else:
                # Initialize session state for checkpoints if it doesn't exist
                if hasattr(st, 'session_state'):
                    st.session_state.checkpoints = {}
                self.session["checkpoint_info"] = f"memory-saver (new): session-based persistence"
                if self.debug:
                    print("[CHECKPOINT] Starting with empty checkpoints")
            
        except Exception as e:
            # Fallback to in-memory if SQLite unavailable
            self.checkpointer = MemorySaver()
            self._checkpointer_cm = None
            self.session["checkpoint_info"] = f"memory-saver fallback (error: {type(e).__name__})"
        
        # Disk-backed per-tid persistence
        self.disk_store = DiskTidStore(session.get("checkpoint_dir", "thread_checkpoints"), debug=self.debug)
        
        self.agent = agent
        self.store = DeepLakePDFStore(commit_batch=8, debug=self.debug)
        # thresholds (configurable via session)
        self.max_turns = session.get("max_turns", 12)
        self.summarize_after_turns = session.get("summarize_after_turns", 12)
        # Load persona template (path can be overridden via session)
        self.persona_template = _read_persona_template(
            session.get("persona_prompt_path", "calum_prompt_v2.1.yaml")
        )
        # Build the graph but defer compilation until first use
        self._graph = None

    async def _ensure_graph(self) -> None:
        """Ensure graph is compiled with checkpointer properly set up."""
        if self._graph is None:
            # Checkpointer is already ready (MemorySaver)
            self._graph = self._build_graph()

    @property
    def graph(self):
        """Get the compiled graph."""
        if self._graph is None:
            raise RuntimeError("Graph not ready. Call await _ensure_graph() first.")
        return self._graph

    def _summarize_history_sync(self, history: List[str]) -> str:
        """Sync wrapper for summarization call (used in to_thread)."""
        if not history:
            return ""
        # Small instructive prompt to summarizer LLM
        prompt = (
            f"You are {self.session.get('persona_name','Calum Worthy')}. "
            f"Summarize this conversation in 1-2 witty sentences especially remembering peculiar things:\n{' | '.join(history)}"
        )
        try:
            client = openai.OpenAI(
                api_key=self.session["api_key"],
                base_url=self.session.get("base_url", "https://api.openai.com/v1"),
            )
            # Wrap client for LangSmith tracing
            client = wrap_openai(client)
            resp = client.chat.completions.create(
                model=self.session.get("model_name", "gpt-4o"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return ""

    async def _summarize_history(self, history: List[str]) -> str:
        return await asyncio.to_thread(self._summarize_history_sync, history)

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AgentState)

        async def write_context_node(state: AgentState) -> AgentState:
            if self.debug:
                print("[NODE] write_context enter")
                print(f"[NODE] Incoming state keys: {list(state.keys())}")
                existing_scratchpad = state.get("scratchpad", [])
                print(f"[NODE] Existing scratchpad length: {len(existing_scratchpad)}")
                if existing_scratchpad:
                    print(f"[NODE] Existing scratchpad: {existing_scratchpad}")
            stt = dict(state)
            user_input = stt.get("user_input")
            if not user_input:
                return stt
            
            # Get existing scratchpad and append new entry
            existing_scratchpad = stt.get("scratchpad", []) or []
            new_entry = f"User: {user_input}"
            
            # Simple deduplication: don't add if it's identical to the last entry
            if not existing_scratchpad or existing_scratchpad[-1] != new_entry:
                new_scratchpad = list(existing_scratchpad) + [new_entry]
                stt["scratchpad"] = new_scratchpad
                if self.debug:
                    print(f"[NODE] Added user entry, new length: {len(new_scratchpad)}")
            else:
                if self.debug:
                    print(f"[NODE] Skipped duplicate user entry")
            
            if self.debug:
                print("[NODE] write_context exit")
            return stt

        def select_context_node(state: AgentState) -> AgentState:
            if self.debug:
                print("[NODE] select_context enter")
            st = dict(state)
            user_input = st.get("user_input", "")
            scratchpad_entries = st.get("scratchpad", [])[-8:]
            st["selected_context"] = {
                "memories_query": user_input,
                "recent_turns": scratchpad_entries
            }
            if self.debug:
                print("[NODE] select_context exit")
            return st

        async def resolve_context_node(state: AgentState) -> AgentState:
            if self.debug:
                print("[NODE] resolve_context enter")
            st = dict(state)
            sel = st.get("selected_context", {})
            q = sel.get("memories_query", "")
            
            # Fast path: try to get recent memories from cache or use simplified context
            try:
                # Quick memory retrieval with timeout
                memories_task = asyncio.create_task(self.store.rag_query(q, top_k=3, agent_id_filter="1"))
                memories = await asyncio.wait_for(memories_task, timeout=1.0)  # 1 second timeout
            except asyncio.TimeoutError:
                if self.debug:
                    print("[NODE] Memory query timed out, using fallback")
                memories = []  # Fallback to empty memories for fast response
                # Continue memory query in background
                asyncio.create_task(self._background_memory_query(q))
            except Exception as e:
                if self.debug:
                    print(f"[NODE] Memory query error: {e}")
                memories = []
            
            recent = sel.get("recent_turns", [])
            st["selected_context"] = f"Memories: {' | '.join(memories)}\nRecent: {' | '.join(recent)}"
            st["top_memory"] = memories[0] if memories else ""
            
            # Add user input as memory immediately (but async to avoid blocking)
            asyncio.create_task(self.store.add_memory(self.agent, f"User: {q}"))
            
            if self.debug:
                print("[NODE] resolve_context exit")
            return st

        async def compress_context_node(state: AgentState) -> AgentState:
            if self.debug:
                print("[NODE] compress_context enter")
            st = dict(state)
            scratchpad = st.get("scratchpad", []) or []
            if len(scratchpad) >= self.summarize_after_turns:
                summary = await self._summarize_history(scratchpad)
                if summary:
                    if self.debug:
                        print(f"[NODE] compress_context: summarizing {len(scratchpad)} entries")
                    # Add summary to memory store
                    asyncio.create_task(self.store.add_memory(self.agent, f"Summary: {summary}"))
                    st["compressed_history"] = summary
                    # Note: We don't truncate the scratchpad here with operator.add
                    # The truncation will be handled during persistence
            if self.debug:
                print("[NODE] compress_context exit")
            return st

        def isolate_context_node(state: AgentState) -> AgentState:
            if self.debug:
                print("[NODE] isolate_context enter/exit")
            stt = dict(state)
            selected_context = stt.get("selected_context", "")
            compressed_history = stt.get("compressed_history", "")
            # Build system prompt from persona template + injected context
            system_prompt = self._build_system_prompt(selected_context, compressed_history)
            # Add debug print for context
            if self.debug:
                print("[DL] LLM system prompt built.")
            stt["agent_context"] = system_prompt
            return stt

        async def llm_node(state: AgentState) -> AgentState:
            if self.debug:
                print("[NODE] llm enter")
                existing_scratchpad = state.get("scratchpad", [])
                print(f"[NODE] LLM - Incoming scratchpad length: {len(existing_scratchpad)}")
            stt = dict(state)
            # Otherwise, call LLM with context
            try:
                messages = [
                    {"role": "system", "content": stt.get("agent_context", "")},
                    {"role": "user", "content": stt.get("user_input", "")}
                ]
                def _call_llm():
                    client = openai.OpenAI(
                        api_key=self.session["api_key"],
                        base_url=self.session.get("base_url", "https://api.openai.com/v1"),
                    )
                    client = wrap_openai(client)
                    return client.chat.completions.create(
                        model=self.session.get("model_name", "gpt-4o"),
                        messages=messages,
                        temperature=self.session.get("temperature", 0.3)
                    )
                resp = await asyncio.to_thread(_call_llm)
                answer = resp.choices[0].message.content.strip()
                
                # Get existing scratchpad and append assistant response
                existing_scratchpad = stt.get("scratchpad", []) or []
                new_entry = f"{self.session.get('persona_name','Calum')}: {answer}"
                
                # Simple deduplication: don't add if it's identical to the last entry
                if not existing_scratchpad or existing_scratchpad[-1] != new_entry:
                    new_scratchpad = list(existing_scratchpad) + [new_entry]
                    stt["scratchpad"] = new_scratchpad
                    if self.debug:
                        print(f"[NODE] LLM - Added assistant entry, new length: {len(new_scratchpad)}")
                    asyncio.create_task(self.store.add_memory(self.agent, new_entry))
                else:
                    if self.debug:
                        print(f"[NODE] LLM - Skipped duplicate assistant entry")
                
                stt["response"] = answer
            except Exception as e:
                if self.debug:
                    print("[LLM ERROR]", repr(e))
                    traceback.print_exc()
                stt["response"] = "Oops — something went wrong. Try again?"
            if self.debug:
                print("[NODE] llm exit")
            return stt

        # register nodes (mix sync & async nodes; LangGraph will handle)
        graph.add_node("write_context", write_context_node)
        graph.add_node("select_context", select_context_node)
        graph.add_node("resolve_context", resolve_context_node)
        graph.add_node("compress_context", compress_context_node)
        graph.add_node("isolate_context", isolate_context_node)
        graph.add_node("llm", llm_node)

        # edges
        graph.add_edge("write_context", "select_context")
        graph.add_edge("select_context", "resolve_context")
        graph.add_edge("resolve_context", "compress_context")
        graph.add_edge("compress_context", "isolate_context")
        graph.add_edge("isolate_context", "llm")
        graph.add_edge("llm", END)

        graph.set_entry_point("write_context")
        return graph.compile(checkpointer=self.checkpointer)

    def _extract_scratchpad_from_checkpoint(self, checkpoint: Any) -> List[str]:
        """Safely extract scratchpad list from a MemorySaver checkpoint object/dict."""
        try:
            # Many versions return dict-like with key 'values' -> {'channel_values': {...}}
            if isinstance(checkpoint, dict):
                values = checkpoint.get("values", {})
            else:
                values = getattr(checkpoint, "values", {})
                if not isinstance(values, dict):
                    values = {}
            channel_values = values.get("channel_values", {})
            scratchpad = channel_values.get("scratchpad", [])
            return scratchpad if isinstance(scratchpad, list) else []
        except Exception:
            return []

    @traceable(name="conversation_turn")
    async def run_turn(self, user_input: str, state: Optional[AgentState] = None) -> AgentState:
        if self.debug:
            print(f"[TURN] user_input='{user_input}'")
        
        # Ensure graph is ready with async checkpointer
        await self._ensure_graph()
        
        # Thread id controls which conversation is resumed
        thread_id = self.session.get("thread_id", f"agent-{self.agent}")
        config = {"configurable": {"thread_id": thread_id}}
        
        if self.debug:
            print(f"[TURN] Using thread_id: {thread_id}")
        
        # Check if checkpointer already has state for this thread
        has_checkpoint = False
        try:
            existing_state = self.checkpointer.get(config)
            has_checkpoint = existing_state is not None
            if self.debug:
                print(f"[TURN] Checkpoint exists: {has_checkpoint}")
        except Exception as e:
            if self.debug:
                print(f"[TURN] Error checking checkpoint: {e}")
            has_checkpoint = False
        
        # Only seed from disk if no checkpoint exists
        input_state: AgentState = {"user_input": user_input}
        if not has_checkpoint:
            disk_seed = self._load_thread_from_disk(thread_id)
            if disk_seed.get("scratchpad"):
                input_state.update(disk_seed)
                if self.debug:
                    print(f"[TURN] Seeded from disk: {len(disk_seed.get('scratchpad', []))} entries")
                    print(f"[TURN] Input state scratchpad: {disk_seed.get('scratchpad', [])}")
        else:
            if self.debug:
                print("[TURN] Using existing checkpoint, not seeding from disk")
        
        if self.debug:
            print(f"[TURN] About to invoke graph with input_state keys: {list(input_state.keys())}")
            if 'scratchpad' in input_state:
                print(f"[TURN] Input scratchpad length: {len(input_state['scratchpad'])}")
        
        # Let LangGraph handle state restoration via checkpointer
        final_state = await self.graph.ainvoke(input_state, config=config)
        
        if self.debug:
            print(f"[TURN] Final state scratchpad length: {len(final_state.get('scratchpad', []))}")
        
        # Persist to disk for this tid
        self._persist_thread_to_disk(thread_id, final_state)
        
        if self.session.get("force_sync_flush"):
            if self.debug:
                print("[TURN] force sync flush")
            await self.store.flush()
        else:
            asyncio.create_task(self.store.flush())
        self._save_checkpoints_to_session()
        return final_state

    @traceable(name="conversation_turn_fast")
    async def run_turn_fast(self, user_input: str, state: Optional[AgentState] = None) -> AgentState:
        """Fast version that returns response immediately and handles memory operations in background."""
        if self.debug:
            print(f"[TURN_FAST] user_input='{user_input}'")
        
        # Ensure graph is ready with async checkpointer
        await self._ensure_graph()
        
        # Thread id controls which conversation is resumed
        thread_id = self.session.get("thread_id", f"agent-{self.agent}")
        config = {"configurable": {"thread_id": thread_id}}
        
        if self.debug:
            print(f"[TURN_FAST] Using thread_id: {thread_id}")
        
        # Check if checkpointer already has state for this thread
        has_checkpoint = False
        try:
            existing_state = self.checkpointer.get(config)
            has_checkpoint = existing_state is not None
            if self.debug:
                print(f"[TURN_FAST] Checkpoint exists: {has_checkpoint}")
        except Exception as e:
            if self.debug:
                print(f"[TURN_FAST] Error checking checkpoint: {e}")
            has_checkpoint = False
        
        # Only seed from disk if no checkpoint exists
        input_state: AgentState = {"user_input": user_input}
        if not has_checkpoint:
            disk_seed = self._load_thread_from_disk(thread_id)
            if disk_seed.get("scratchpad"):
                input_state.update(disk_seed)
                if self.debug:
                    print(f"[TURN_FAST] Seeded from disk: {len(disk_seed.get('scratchpad', []))} entries")
                    print(f"[TURN_FAST] Input state scratchpad: {disk_seed.get('scratchpad', [])}")
        else:
            if self.debug:
                print("[TURN_FAST] Using existing checkpoint, not seeding from disk")
        
        if self.debug:
            print(f"[TURN_FAST] About to invoke graph with input_state keys: {list(input_state.keys())}")
            if 'scratchpad' in input_state:
                print(f"[TURN_FAST] Input scratchpad length: {len(input_state['scratchpad'])}")
        
        # Let LangGraph handle state restoration via checkpointer
        final_state = await self.graph.ainvoke(input_state, config=config)
        
        if self.debug:
            print(f"[TURN_FAST] Final state scratchpad length: {len(final_state.get('scratchpad', []))}")
        
        # Persist to disk for this tid
        self._persist_thread_to_disk(thread_id, final_state)
        
        # Handle background operations asynchronously (non-blocking)
        asyncio.create_task(self._background_post_turn_operations())
        self._save_checkpoints_to_session()
        return final_state

    async def _background_post_turn_operations(self):
        """Handle background operations after response is returned."""
        try:
            if self.debug:
                print("[BACKGROUND] Running post-turn operations...")
            # Just flush memory operations - the actual memory additions happen during graph execution
            await self.store.flush()
            if self.debug:
                print("[BACKGROUND] Post-turn operations completed")
        except Exception as e:
            if self.debug:
                print(f"[BACKGROUND] Error in post-turn operations: {e}")

    async def _background_memory_query(self, query: str):
        """Run memory query in background for future use."""
        try:
            if self.debug:
                print(f"[BACKGROUND] Running delayed memory query for: {query[:50]}...")
            await self.store.rag_query(query, top_k=5, agent_id_filter="1")
            if self.debug:
                print("[BACKGROUND] Delayed memory query completed")
        except Exception as e:
            if self.debug:
                print(f"[BACKGROUND] Delayed memory query error: {e}")

    def _save_checkpoints_to_session(self):
        """Save checkpoints to Streamlit session state for persistence across reloads."""
        try:
            if hasattr(st, 'session_state') and hasattr(self.checkpointer, 'storage'):
                # Make sure we save a copy of the storage dict
                st.session_state.checkpoints = dict(self.checkpointer.storage)
                if self.debug:
                    print(f"[CHECKPOINT] Saved {len(self.checkpointer.storage)} checkpoints to session")
                    print(f"[CHECKPOINT] Saved checkpoint keys: {list(self.checkpointer.storage.keys())}")
                    
                    # Debug: look at the latest checkpoint content
                    thread_id = self.session.get("thread_id", f"agent-{self.agent}")
                    for key, checkpoint in self.checkpointer.storage.items():
                        key_str = str(key)
                        if thread_id in key_str:
                            try:
                                scratchpad = self._extract_scratchpad_from_checkpoint(checkpoint)
                                print(f"[CHECKPOINT] Checkpoint {key_str} scratchpad length: {len(scratchpad) if scratchpad else 0}")
                                if scratchpad:
                                    print(f"[CHECKPOINT] Checkpoint {key_str} scratchpad: {scratchpad}")
                            except Exception as e:
                                print(f"[CHECKPOINT] Error inspecting checkpoint {key_str}: {e}")
                            break
        except Exception as e:
            if self.debug:
                print(f"[CHECKPOINT] Error saving to session: {e}")

    def _persist_thread_to_disk(self, thread_id: str, state: AgentState):
        """Persist minimal thread state to disk keyed by tid."""
        try:
            scratchpad = state.get("scratchpad", []) or []
            # Trim to last max_turns to keep files small
            scratchpad = scratchpad[-self.max_turns:]
            compressed = state.get("compressed_history", "") or ""
            data = {"scratchpad": scratchpad, "compressed_history": compressed}
            self.disk_store.save(thread_id, data)
        except Exception as e:
            if self.debug:
                print(f"[DISK] Persist error for {thread_id}: {e}")

    def _load_thread_from_disk(self, thread_id: str) -> Dict[str, Any]:
        data = self.disk_store.load(thread_id)
        # Validate types
        sp = data.get("scratchpad", []) if isinstance(data, dict) else []
        if not isinstance(sp, list):
            sp = []
        ch = data.get("compressed_history", "") if isinstance(data, dict) else ""
        if not isinstance(ch, str):
            ch = ""
        if self.debug and sp:
            print(f"[DISK] Seeded scratchpad for {thread_id}: len={len(sp)}")
        return {"scratchpad": sp, "compressed_history": ch}

    # CLI convenience
    async def run_cli(self):
        print(f"{self.session.get('persona_name','Calum Worthy')} - (type 'exit' to quit)")
        # Ensure graph is ready with checkpointer
        await self._ensure_graph()
        try:
            while True:
                user_input = await asyncio.to_thread(input, "You: ")
                if user_input.strip().lower() == "exit":
                    await self.store.flush()
                    break
                new_state = await self.run_turn(user_input)
                # Force flush after first turn to ensure tensors exist
                await self.store.flush()
                print(f"{self.session.get('persona_name','Calum')}: {new_state.get('response','')}")
                # No manual state tracking needed; checkpointer persists it.
        finally:
            await self._cleanup()

    async def _cleanup(self):
        """Clean up resources, especially the checkpointer context manager."""
        if self._checkpointer_cm is not None and self.checkpointer is not None:
            try:
                await self._checkpointer_cm.__aexit__(None, None, None)
            except Exception as e:
                if self.debug:
                    print(f"[CLEANUP] Error closing checkpointer: {e}")

    def _build_system_prompt(self, selected_context: str, compressed_history: str) -> str:
        """Render the persona template with injected context, summary, and current time."""
        tmpl = self.persona_template
        ctx = selected_context or ""
        summ = compressed_history or ""
        now_str = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        # Inject NOW
        if "{NOW}" in tmpl:
            tmpl = tmpl.replace("{NOW}", now_str)
        else:
            # Prepend a Current time line if not provided as a placeholder
            tmpl = f"Current time: {now_str}\n\n" + tmpl
        # Inject CONTEXT
        if "{CONTEXT}" in tmpl:
            tmpl = tmpl.replace("{CONTEXT}", ctx)
        elif "Context:" in tmpl:
            tmpl = tmpl.replace("Context:", f"Context:\n{ctx}\n")
        else:
            tmpl += f"\n\n[Context]\n{ctx}\n"
        # Inject SUMMARY
        if "{SUMMARY}" in tmpl:
            tmpl = tmpl.replace("{SUMMARY}", summ)
        elif "Summary:" in tmpl:
            tmpl = tmpl.replace("Summary:", f"Summary:\n{summ}\n")
        else:
            tmpl += f"\n\n[Summary]\n{summ}\n"
        return tmpl

# ---------- Optional FastAPI adapter (simple) ----------
# app = FastAPI()
# conversational_system = OrchestratedConversationalSystem(session={
#     "api_key": os.getenv("OPENAI_API_KEY"),
#     "model_name": "gpt-4",
#     "base_url": "https://api.openai.com/v1",
#     "persona_name": "Calum",
#     "avatar_id": "calum",
#     "avatar_prompts": {"calum": "You are Calum Worthy, a witty activist and actor."}
# })
#
# @app.post("/chat")
# async def chat_endpoint(payload: Dict[str, str]):
#     user_input = payload.get("user_input", "")
#     if not user_input:
#         raise HTTPException(status_code=400, detail="user_input required")
#     state = AgentState(
#         session=conversational_system.session,
#         scratchpad=[]
#     )
#     new_state = await conversational_system.run_turn(user_input, state)
#     return {"response": new_state.get("response", "")}
#
# if __name__ == "__main__":
#     uvicorn.run("this_module:app", host="0.0.0.0", port=8000, reload=True)

# ---------- Run CLI if invoked directly ----------


if __name__ == "__main__":
    # Set up LangSmith tracing environment variables first
    from dotenv import load_dotenv
    load_dotenv()  # Load from .env file for CLI mode
    
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "calum-worthy-chatbot")
    
    # Only use st.secrets if we're actually in a Streamlit context
    try:
        import streamlit as st
        api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        activeloop_token = st.secrets.get("ACTIVELOOP_TOKEN", os.getenv("ACTIVELOOP_TOKEN", ""))
        activeloop_org = st.secrets.get("ACTIVELOOP_ORG_ID", os.getenv("ACTIVELOOP_ORG_ID", ""))
        langchain_key = st.secrets.get("LANGCHAIN_API_KEY", os.getenv("LANGCHAIN_API_KEY", ""))
    except:
        # Fallback to environment variables when not in Streamlit
        api_key = os.getenv("OPENAI_API_KEY", "")
        activeloop_token = os.getenv("ACTIVELOOP_TOKEN", "")
        activeloop_org = os.getenv("ACTIVELOOP_ORG_ID", "")
        langchain_key = os.getenv("LANGCHAIN_API_KEY", "")
    
    os.environ.setdefault("OPENAI_API_KEY", api_key)
    os.environ.setdefault("ACTIVELOOP_TOKEN", activeloop_token)
    os.environ.setdefault("ACTIVELOOP_ORG_ID", activeloop_org)
    os.environ.setdefault("LANGCHAIN_API_KEY", langchain_key)

    session = {
        "api_key": api_key,
        "model_name": "gpt-4o",
        "base_url": "https://api.openai.com/v1",
        "persona_name": "Calum",
        "avatar_id": "calum",
        "avatar_prompts": {"calum": "You are Calum Worthy, a witty activist and actor."},
        "temperature": 0.3,
        "debug": True,               # turn on verbose debug
        "force_sync_flush": False    # set True to wait every turn
    }
    # Initialize the embedding model BEFORE constructing the system
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=session.get("api_key")
    )
    conv = OrchestratedConversationalSystem(session=session)
    try:
        asyncio.run(conv.run_cli())
    except KeyboardInterrupt:
        asyncio.run(conv.store.flush())
        asyncio.run(conv._cleanup())
        raise






