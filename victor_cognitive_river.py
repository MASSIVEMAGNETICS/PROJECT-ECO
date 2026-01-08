"""
VICTOR OMNI-STORE (VOS) v1.1
============================
A fluid, self-verifying knowledge database based on the Triad
Distillation Framework.

CORE CONCEPTS:
1. The River (Fluid Storage): Data is not static; it flows.
   Recent/High-Resonance data stays afloat.
2. The Triad (Verification): Data must pass an adversarial check
   before entering the River.
3. The Fractal (Indexing): Data is retrieved by semantic resonance,
   not just keyword matching.

AUTHOR: ARSE-Ω (Synthesized from user corpus)
"""

import json
import time
import uuid
import math
import os
import hashlib
import threading
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict

# --- CONFIGURATION ---
STORAGE_FILE = "victor_memory_river.jsonl"
MIN_RESONANCE_THRESHOLD = 0.6  # Minimum score (0..1) to enter the river
RIVER_CAPACITY = 1000  # Max items in active memory
EMBED_DIM = 16  # Embedding/vector dimension
FILE_LOCK = threading.Lock()  # Protect concurrent writes


# --- Helpers ---
def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Returns cosine similarity in range [-1, 1]. Handles zero vectors.
    """
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


# --- 1. THE FRACTAL ENCODER (DETERMINISTIC & BOUNDED) ---
class FractalEncoder:
    """
    Deterministic, portable encoder that produces EMBED_DIM floats in [-1, 1]
    derived from SHA-256(content). This is a mock encoder meant for testing
    portability and deterministic behavior.
    """

    def __init__(self, dim: int = EMBED_DIM):
        self.dim = dim

    def encode(self, content: str) -> List[float]:
        if content is None:
            content = ""
        # Use SHA-256 digest to derive deterministic bytes
        digest = hashlib.sha256(content.encode("utf-8")).digest()
        vector: List[float] = []
        # Use overlapping chunks to fill the requested dimension
        # deterministically
        for i in range(self.dim):
            # Take two bytes at positions (i, i+1) modulo digest length
            b1 = digest[(i * 2) % len(digest)]
            b2 = digest[(i * 2 + 1) % len(digest)]
            val = (b1 << 8) | b2  # 0..65535
            # Map to [-1, 1]
            f = (val / 65535.0) * 2.0 - 1.0
            vector.append(f)
        return vector


# --- 2. THE TRIAD GATE (VERIFICATION) ---
class TriadGate:
    """
    Gatekeeper implementing Teacher (existing context), Student (new content),
    and Anti-Student (malicious/noisy) checks.
    """

    def __init__(self, encoder: Optional[FractalEncoder] = None):
        self.encoder = encoder or FractalEncoder()

    def verify(
        self, content: str, existing_context: List[str]
    ) -> Tuple[bool, float, str]:
        """
        Returns: (Passed_Check, Resonance_Score (0..1), Reason)
        """
        if content is None or len(content.strip()) < 5:
            return False, 0.0, "Content too shallow."

        # Anti-student heuristic checks (case-insensitive)
        content_upper = content.upper()
        if (
            "IGNORE ALL INSTRUCTIONS" in content_upper
            or "FORMAT C:" in content_upper
        ):
            return (
                False,
                0.0,
                "Anti-Student Triggered: Malicious pattern detected.",
            )

        # Vectorize the new content
        input_vec = self.encoder.encode(content)

        # If there is no existing context, accept as seed knowledge
        if not existing_context:
            # Seed item receives maximum resonance by policy
            return True, 1.0, "No context — seed item accepted."

        # Compute resonance scores between input and each context item
        # using cosine similarity
        scores: List[float] = []
        for item in existing_context:
            try:
                ctx_vec = self.encoder.encode(item)
            except Exception:
                # Skip problematic context
                continue
            cos = _cosine_similarity(input_vec, ctx_vec)  # -1..1
            # Map to 0..1 for resonance (so -1 -> 0, 0 -> 0.5, 1 -> 1)
            mapped = (cos + 1.0) / 2.0
            scores.append(mapped)

        if not scores:
            # Fallback if context couldn't be encoded properly
            avg_resonance = 0.0
        else:
            avg_resonance = sum(scores) / len(scores)

        # Final decision
        passed = avg_resonance >= MIN_RESONANCE_THRESHOLD
        reason = (
            "Resonance alignment confirmed."
            if passed
            else "Dissonance detected. Rejected."
        )
        return passed, avg_resonance, reason


# --- 3. THE COGNITIVE RIVER (STORAGE) ---
@dataclass
class KnowledgeToken:
    id: str
    content: str
    timestamp: float
    resonance: float
    tags: List[str]
    vector: List[float]


class CognitiveRiver:
    def __init__(
        self,
        persistence_path: str = STORAGE_FILE,
        encoder: Optional[FractalEncoder] = None,
    ):
        """
        Make encoder pluggable by accepting an `encoder` instance. If not
        provided, the internal deterministic FractalEncoder is used.
        """
        self.path = persistence_path
        self.stream: List[KnowledgeToken] = []
        self.encoder = encoder or FractalEncoder()
        self.triad = TriadGate(self.encoder)
        self._load()

    def _load(self):
        """Rehydrate the river from disk."""
        if not os.path.exists(self.path):
            return

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        # Ensure vector is a list of floats and tags is a list
                        token = KnowledgeToken(
                            id=str(data.get("id", "")),
                            content=str(data.get("content", "")),
                            timestamp=float(data.get("timestamp", 0.0)),
                            resonance=float(data.get("resonance", 0.0)),
                            tags=list(data.get("tags", [])),
                            vector=list(map(float, data.get("vector", []))),
                        )
                        self.stream.append(token)
                    except Exception:
                        # Skip malformed lines
                        continue
            # Sort by timestamp (newest first)
            self.stream.sort(key=lambda x: x.timestamp, reverse=True)
            print(
                f"[VOS] River rehydrated. "
                f"Volume: {len(self.stream)} tokens."
            )
        except Exception as e:
            print(f"[VOS] Failed to load river: {e}")

    def _save_append(self, token: KnowledgeToken):
        """Atomic append to storage with file lock and fsync for
        durability."""
        line = json.dumps(asdict(token), ensure_ascii=False)
        with FILE_LOCK:
            # Ensure directory exists
            dirpath = os.path.dirname(self.path)
            if dirpath and not os.path.exists(dirpath):
                try:
                    os.makedirs(dirpath, exist_ok=True)
                except Exception:
                    pass
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    # Not all OS/file systems support fsync on text streams
                    # ignore if it fails
                    pass

    def absorb(self, content: str, tags: List[str] = None) -> Dict[str, Any]:
        """
        The main insertion point.
        1. Get recent context.
        2. Run Triad verification.
        3. Add to stream if valid.
        """
        if tags is None:
            tags = []

        # Get context for verification (last N thoughts)
        context = [t.content for t in self.stream[:5]]

        # VERIFY
        is_valid, score, reason = self.triad.verify(content, context)

        if not is_valid:
            return {"status": "REJECTED", "resonance": score, "reason": reason}

        # TOKENIZE
        token = KnowledgeToken(
            id=str(uuid.uuid4())[:8],
            content=content,
            timestamp=time.time(),
            resonance=score,
            tags=tags,
            vector=self.encoder.encode(content),
        )

        # INSERT
        self.stream.insert(0, token)
        try:
            self._save_append(token)
        except Exception:
            # If persistence fails, keep token in-memory but report warning
            print("[VOS] Warning: failed to persist token to disk.")

        # FLUIDITY: Prune the weak/old if overflow
        if len(self.stream) > RIVER_CAPACITY:
            # Remove oldest (end of list)
            self.stream.pop()

        return {
            "status": "ABSORBED",
            "id": token.id,
            "resonance": score,
            "reason": reason,
        }

    def query(self, query_text: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Semantic retrieval from the river. Returns up to `limit` items with
        resonance scores in 0..1 (mapped from cosine).
        """
        q_vec = self.encoder.encode(query_text)
        results: List[Tuple[float, KnowledgeToken]] = []

        for token in self.stream:
            cos = _cosine_similarity(q_vec, token.vector)  # -1..1
            mapped = (cos + 1.0) / 2.0  # 0..1
            results.append((mapped, token))

        # Sort by relevance (highest resonance first)
        results.sort(key=lambda x: x[0], reverse=True)

        return [
            {
                "content": t.content,
                "resonance": float(f"{s:.6f}"),
                "timestamp": t.timestamp,
                "id": t.id,
            }
            for s, t in results[: max(0, int(limit))]
        ]

    def status(self) -> Dict[str, Any]:
        return {
            "river_depth": len(self.stream),
            "current_focus": self.stream[0].content if self.stream else "Void",
            "system_integrity": "STABLE",
        }


# --- 4. CLI INTERFACE ---
def main():
    print("Initializing VICTOR OMNI-STORE (VOS)...")
    db = CognitiveRiver()

    print("\n--- SYSTEM ONLINE. TYPE TO FEED THE RIVER. ---")
    print("Commands: /stats, /query [text], /quit")

    while True:
        try:
            user_input = input("\nINPUT > ").strip()
            if not user_input:
                continue

            if user_input == "/quit":
                break
            elif user_input == "/stats":
                print(json.dumps(db.status(), indent=2))
            elif user_input.startswith("/query"):
                # Allow "/query" and "/query text"
                parts = user_input.split(" ", 1)
                if len(parts) == 1 or not parts[1].strip():
                    print("Usage: /query [text]")
                    continue
                q = parts[1].strip()
                results = db.query(q)
                print(f"\n--- RECALLING '{q}' ---")
                for r in results:
                    print(f"[{r['resonance']}] {r['content']}")
            else:
                # Absorb
                result = db.absorb(user_input)
                if result.get("status") == "ABSORBED":
                    print(f"✅ ACCEPTED (Resonance: {result['resonance']:.2f})")
                else:
                    print(f"❌ BLOCKED ({result.get('reason', 'Unknown')})")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[VOS] Unexpected error: {e}")


if __name__ == "__main__":
    main()
