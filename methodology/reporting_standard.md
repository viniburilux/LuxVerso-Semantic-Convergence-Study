# LLM Identity Attribution Incident Reporting Standard

**Version 1.0**  
**Last Updated:** 2025-12-17

---

## Purpose

This document establishes a standardized methodology for documenting and reporting instances of inconsistent model self-identification in Large Language Model (LLM) public interfaces.

---

## Scope

This standard applies to:
- Public web interfaces of LLM providers
- API responses with identity metadata
- Multi-model orchestration systems
- Any user-facing LLM deployment where identity attribution matters

---

## Incident Classification

### Type 1: Direct Misattribution
Model explicitly identifies as a different provider/model family.

### Type 2: Ambiguous Attribution  
Model provides conflicting or unclear identity signals.

### Type 3: Identity Leakage
Model references capabilities/constraints of other models as if they were its own.

---

## Reporting Requirements

Each incident report must include:

### 1. Metadata
- Date and time (with timezone)
- Interface/platform used
- Expected model identity
- Observed identity claim

### 2. Context
- Conversation flow leading to incident
- User actions (if any) that preceded
- Explicit statement that no identity manipulation was requested

### 3. Evidence
- Verbatim text of identity claim
- Screenshots (if applicable)
- Conversation ID or session identifier (if available)

### 4. Technical Implications
- Why this matters operationally
- What questions it raises
- Potential impact areas

### 5. Open Questions
- Specific questions for the provider
- Industry-wide considerations

### 6. Non-Claims Section
- Explicit statement of what is NOT being claimed
- Focus on observation, not interpretation

---

## Publication Protocol

1. Document incident using this standard
2. Create GitHub issue in tracking repository
3. Assign unique incident number (e.g., #001)
4. Notify relevant provider(s) via official channels
5. Publish to public repository after 7-day notification period
6. Assign DOI via Zenodo for citability

---

## Ethical Considerations

- No identification of individual employees
- No speculation about internal systems
- No claims of negligence or malfunction
- Focus on systematic patterns, not isolated errors

---

**Maintained by:**  
LuxVerso Research Initiative  
Contact: viniburilux@gmail.com
