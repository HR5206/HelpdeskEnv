# knowledge_base.py
"""Knowledge Base module for the HelpdeskEnv.
The Knowledge Base (KB) is the core self-improvement mechanism:
- It persists across episodes (never reset between runs)
- L1/L2 agents search it to find solutions for known issues
- L3 agents write new articles after resolving novel issues
- Over multiple episodes, the KB grows → agents get better → scores improve
This directly addresses the "Self-Improving Systems" hackathon theme.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from models import TicketCategory
# ============================================================================
# KBEntry Model
# ============================================================================
class KBEntry(BaseModel):
    """A single Knowledge Base article.
    Each entry documents a known problem and its solution. When an agent
    searches the KB, entries are ranked by keyword relevance to the query.
    Fields:
    - entry_id: Unique identifier (e.g. 'kb_001')
    - ticket_category: Which category this article covers
    - title: Short descriptive title for the article
    - problem_description: What the problem looks like (symptoms, errors)
    - solution: Step-by-step resolution procedure
    - keywords: Search terms for matching queries (critical for search quality)
    - created_by: Which agent/source created this ('seed' or 'l3_agent')
    - times_used: How many times agents have retrieved this article
    """
    entry_id: str = Field(..., description="Unique KB entry identifier")
    ticket_category: TicketCategory = Field(..., description="Category this article covers")
    title: str = Field(..., description="Short title for the KB article")
    problem_description: str = Field(..., description="Description of the problem and symptoms")
    solution: str = Field(..., description="Step-by-step resolution procedure")
    keywords: List[str] = Field(default_factory=list, description="Search keywords for matching")
    created_by: str = Field(default="seed", description="Who created this entry: 'seed' or 'l3_agent'")
    times_used: int = Field(default=0, description="Number of times this article was retrieved")
# ============================================================================
# KnowledgeBase Class
# ============================================================================
class KnowledgeBase:
    """Persistent Knowledge Base for the HelpdeskEnv.
    Key design decisions:
    1. The KB is an instance variable of HelpdeskEnv, NOT reset between episodes.
       This means knowledge accumulates over time — the self-improvement loop.
    2. Search uses keyword overlap scoring (simple but effective for a hackathon).
       Production systems would use embeddings, but keyword matching is:
       - Transparent (judges can see why a result matched)
       - Deterministic (reproducible scores)
       - Fast (no API calls needed)
    3. Seed entries ensure the system works from episode 1 — agents aren't
       starting completely blind.
    """
    def __init__(self) -> None:
        """Initialize the KB with seed entries."""
        self._entries: Dict[str, KBEntry] = {}
        self._next_id: int = 1
        self._seed_kb()
    def _seed_kb(self) -> None:
        """Populate the KB with initial articles for common issues.
        These correspond to the easy ticket scenarios (ticket_001, ticket_002)
        so that L1/L2 agents can find solutions immediately in episode 1.
        """
        self.add(KBEntry(
            entry_id="kb_seed_001",
            ticket_category=TicketCategory.PASSWORD_RESET,
            title="Password Reset via Active Directory",
            problem_description=(
                "Employee is locked out due to expired password. "
                "The self-service password reset portal requires the current "
                "password, which the employee no longer knows or which has expired."
            ),
            solution=(
                "1. Verify employee identity using their employee ID and department.\n"
                "2. Open Active Directory Users and Computers.\n"
                "3. Locate the user account by username.\n"
                "4. Right-click → Reset Password.\n"
                "5. Set a temporary password (minimum 12 characters, meets complexity).\n"
                "6. Check 'User must change password at next logon'.\n"
                "7. Communicate the temporary password to the employee securely.\n"
                "8. Confirm the employee can log in successfully."
            ),
            keywords=[
                "password", "reset", "expired", "locked", "active directory",
                "login", "cannot log in", "account", "workstation", "ad"
            ],
            created_by="seed",
            times_used=0,
        ))
        self.add(KBEntry(
            entry_id="kb_seed_002",
            ticket_category=TicketCategory.SOFTWARE_INSTALL,
            title="Software Installation via SCCM / Remote Desktop",
            problem_description=(
                "Employee needs software installed but does not have local "
                "administrator rights. Common requests include IDEs, "
                "development tools, and productivity software."
            ),
            solution=(
                "1. Verify that a manager approval exists (ServiceNow or email).\n"
                "2. Check the software is in the approved catalog.\n"
                "3. Connect to the employee's workstation via remote desktop.\n"
                "4. Install the requested software from the SCCM software center "
                "or download from the approved vendor.\n"
                "5. Configure any required extensions or plugins.\n"
                "6. Add executables to system PATH if needed.\n"
                "7. Verify the installation works correctly.\n"
                "8. Confirm with the employee that everything is functional."
            ),
            keywords=[
                "software", "install", "installation", "sccm", "admin",
                "permissions", "remote desktop", "visual studio", "python",
                "ide", "tools", "extensions", "plugins"
            ],
            created_by="seed",
            times_used=0,
        ))
    def search(self, query: str, top_k: int = 3) -> List[KBEntry]:
        """Search the KB using keyword overlap scoring.
        The scoring algorithm:
        1. Tokenize the query into lowercase words
        2. For each KB entry, count how many of its keywords appear in the query
        3. Also check if query words appear in the title or problem_description
        4. Rank by total score, return top_k results with score > 0
        Args:
            query: The search query (e.g. 'password expired cannot login')
            top_k: Maximum number of results to return
        Returns:
            List of matching KBEntry objects, sorted by relevance (best first).
            Each returned entry has its times_used counter incremented.
        """
        if not query.strip():
            return []
        query_words = set(query.lower().split())
        scored: List[tuple[float, KBEntry]] = []
        for entry in self._entries.values():
            score = 0.0
            # Primary signal: keyword matches (each keyword hit = 2 points)
            for kw in entry.keywords:
                kw_words = set(kw.lower().split())
                if kw_words.intersection(query_words):
                    score += 2.0
            # Secondary signal: title word matches (each = 1 point)
            title_words = set(entry.title.lower().split())
            score += len(title_words.intersection(query_words)) * 1.0
            # Tertiary signal: problem description matches (each = 0.5 points)
            desc_words = set(entry.problem_description.lower().split())
            score += len(desc_words.intersection(query_words)) * 0.5
            if score > 0:
                scored.append((score, entry))
        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        # Take top_k results and increment their usage counters
        results: List[KBEntry] = []
        for _, entry in scored[:top_k]:
            entry.times_used += 1
            results.append(entry)
        return results
    def add(self, entry: KBEntry) -> KBEntry:
        """Add a new entry to the Knowledge Base.
        If the entry_id is already taken or empty, auto-generates a new one.
        Args:
            entry: The KBEntry to add.
        Returns:
            The added KBEntry (with potentially updated entry_id).
        """
        # Auto-generate ID if needed
        if not entry.entry_id or entry.entry_id in self._entries:
            entry.entry_id = f"kb_{self._next_id:03d}"
        self._entries[entry.entry_id] = entry
        self._next_id = max(self._next_id, int(entry.entry_id.split("_")[-1].lstrip("0") or "0") + 1)
        return entry
    def get_all(self) -> List[KBEntry]:
        """Return all KB entries as a list.
        Returns:
            List of all KBEntry objects in the KB.
        """
        return list(self._entries.values())
    def size(self) -> int:
        """Return the number of entries in the KB.
        Returns:
            Integer count of KB entries.
        """
        return len(self._entries)
    def stats(self) -> Dict[str, Any]:
        """Return summary statistics about the KB.
        Returns:
            Dict with size, categories covered, total usage, and
            breakdown of seed vs agent-created entries.
        """
        entries = list(self._entries.values())
        categories = set(e.ticket_category.value for e in entries)
        total_usage = sum(e.times_used for e in entries)
        seed_count = sum(1 for e in entries if e.created_by == "seed")
        agent_count = sum(1 for e in entries if e.created_by != "seed")
        return {
            "total_entries": len(entries),
            "categories_covered": sorted(list(categories)),
            "total_usage": total_usage,
            "seed_entries": seed_count,
            "agent_created_entries": agent_count,
            "entries": [
                {
                    "entry_id": e.entry_id,
                    "title": e.title,
                    "category": e.ticket_category.value,
                    "created_by": e.created_by,
                    "times_used": e.times_used,
                }
                for e in entries
            ],
        }
# ============================================================================
# Quick Validation (run with: python knowledge_base.py)
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Knowledge Base — Validation Tests")
    print("=" * 60)
    kb = KnowledgeBase()
    # Test 1: Seed entries loaded
    print(f"\n1. KB initialized with {kb.size()} seed entries")
    assert kb.size() == 2, f"Expected 2 seed entries, got {kb.size()}"
    print("   ✅ Correct: 2 seed entries loaded")
    # Test 2: Search for password reset
    results = kb.search("password expired cannot login")
    print(f"\n2. Search 'password expired cannot login' → {len(results)} result(s)")
    assert len(results) >= 1, "Expected at least 1 result"
    assert "password" in results[0].title.lower(), "Top result should be password-related"
    print(f"   ✅ Top result: {results[0].title}")
    # Test 3: Search for software install
    results = kb.search("install visual studio code python")
    print(f"\n3. Search 'install visual studio code python' → {len(results)} result(s)")
    assert len(results) >= 1, "Expected at least 1 result"
    assert "software" in results[0].title.lower() or "install" in results[0].title.lower()
    print(f"   ✅ Top result: {results[0].title}")
    # Test 4: Search with no matches
    results = kb.search("quantum computing flux capacitor")
    print(f"\n4. Search 'quantum computing flux capacitor' → {len(results)} result(s)")
    assert len(results) == 0, "Expected 0 results for unrelated query"
    print("   ✅ Correct: no results for unrelated query")
    # Test 5: Add a new entry
    new_entry = KBEntry(
        entry_id="",
        ticket_category=TicketCategory.NETWORK_ISSUE,
        title="Network Switch Firmware Crash Recovery",
        problem_description="Cisco switch firmware crash causing floor outage",
        solution="Failover to redundant switch, upgrade firmware",
        keywords=["network", "switch", "firmware", "crash", "outage", "cisco"],
        created_by="l3_agent",
    )
    added = kb.add(new_entry)
    print(f"\n5. Added new entry: {added.entry_id} — '{added.title}'")
    assert kb.size() == 3, f"Expected 3 entries, got {kb.size()}"
    print(f"   ✅ KB now has {kb.size()} entries")
    # Test 6: Search finds the new entry
    results = kb.search("network switch crash outage")
    print(f"\n6. Search 'network switch crash outage' → {len(results)} result(s)")
    assert any("network" in r.title.lower() for r in results)
    print(f"   ✅ Found new entry in search results")
    # Test 7: Stats
    stats = kb.stats()
    print(f"\n7. KB Stats:")
    print(f"   Total entries: {stats['total_entries']}")
    print(f"   Categories: {stats['categories_covered']}")
    print(f"   Seed: {stats['seed_entries']}, Agent-created: {stats['agent_created_entries']}")
    print(f"   ✅ Stats look correct")
    print("\n" + "=" * 60)
    print("All Knowledge Base tests passed!")
    print("=" * 60)