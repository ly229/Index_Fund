---
name: thesis-writing
description: Draft and revise thesis sections in a formal economics/finance dissertation style using the reference sample in references/reference_style.md. Use when writing introductions, literature reviews, empirical chapters, abstracts, or figure captions that must mirror the tone, structure, and formatting conventions of the provided reference.
---

# Thesis Writing

## Overview
Produce thesis-quality prose that mirrors the reference dissertation sample (Lu Yu, 2022) with formal, data-driven academic tone suitable for economics/finance.

## Quick Start
- Confirm section type (introduction, literature review, empirical design, results, conclusion) and key inputs (question, data, findings).
- Skim `references/reference_style.md` to refresh tone: formal, third-person, precise wording, explicit transitions, quantified motivation.
- Outline: context and fact → research question → mechanism/intuitions → contributions/approach → roadmap.
- Draft with disciplined transitions ("while", "whereas", "as a result"), author-year citations for literature, numeric footnotes only for clarifications/data sources.
- Reference figures/tables in-text ("Figure 1 shows ...") and supply concise captions that include data source.
- Revise for cohesion: consistent terminology, parallel phrasing, minimal rhetorical flourish, explicit logical flow.

## Style Guardrails (drawn from reference)
- Tone: Objective, analytical, formal; avoid colloquial verbs and adjectives.
- Voice: Mainly third-person; first-person plural acceptable for method statements.
- Sentence shape: Multi-clause but clear; keep connectors explicit; avoid fragments.
- Evidence: Pair claims with statistics or citations; surface data origin in captions or footnotes.
- Citations: Author-year for literature (e.g., "Kim et al. (2012)"); superscript footnotes for data or explanatory notes.
- Captions: Start "Figure X:" followed by a plain summary; end with "Data Source: ..." when applicable.

## Section Patterns
- Introduction: Market fact → question → mechanism intuition → empirical strategy preview → headline finding → roadmap.
- Literature Review: Group by mechanism/theme; contrast findings; identify gaps addressed here.
- Empirical Design: Define sample, variables, identification; state expectations before results.
- Results: Lead with main effect, interpret magnitude, note robustness/alternative explanations.
- Conclusion: Restate contribution and implications; suggest future work briefly.

## Chapter Playbook (from reference chapters)
- Chapter 0 (Cover + Introduction + Research Question): Open with long-run market trend and concrete magnitudes (assets, flows, market share), then state the core question as a market-quality tradeoff (transparency, liquidity, volatility). Explain competing mechanisms before presenting baseline directional findings. Close with a direct research-question subsection listing proxies and a short causal-identification preview.
- Chapter 1 (Literature Review): Organize by outcome channel, not by paper chronology. Use balanced framing: studies finding positive, negative, and ambiguous effects. Explicitly mark measurement disputes (for example, how to measure price informativeness) and position your proxy choice as a contribution.
- Chapter 2 (Data + Filtering): Document each dataset with provider, coverage horizon, frequency, and role in construction. Flag frequency mismatches and state mitigation choices (lags, aggregation window). For sample-construction filters, report validation with false-positive/false-negative checks and a short interpretation of filter quality.
- Chapter 3 (Preliminary + Regression Analysis): Introduce baseline regression as correlation evidence first, then define model equation, variables, and control rationale. Provide formula-level definitions for each key variable (synchronicity transform, liquidity proxies, volatility, ownership share). In results prose, map each column to economic interpretation and call out contradictions or insignificance explicitly instead of smoothing them away.
- Chapter 4 (Future Work + Identification): Separate causal roadmap from panel-extension roadmap. For causal claims, spell out endogeneity source, instrument candidate, relevance logic, and exclusion-risk checks. For panel extension, justify horizon choice with specific stress episodes and explain what heterogeneity test the panel is intended to identify.
- Chapter 5 (References): Keep bibliography broad across theory, empirical evidence, and methods tied to each channel (informativeness, liquidity, volatility, identification). Ensure citation style is consistent with author-year in text and complete reference details in bibliography.

## Chapter-Specific Output Templates
- If user asks for an introduction chapter: Draft in four blocks: market evolution facts; mechanism tension; baseline empirical preview; precise research question and proxy definitions.
- If user asks for a literature section: Draft in subsections by mechanism/outcome with explicit "evidence split" sentences (positive vs negative vs mixed) and one paragraph on measurement/identification gaps.
- If user asks for data/method section: Include data provenance table-style prose (source, frequency, period, variables), then sample/filter validation and model specification with notation.
- If user asks for results section: Start with one-sentence headline per dependent variable group, then magnitude/sign/statistical significance, then unresolved puzzle paragraph.
- If user asks for future work or limitations: Include one subsection on causal identification design and one on panel/time-variation tests with concrete event windows.

## Required Elements Checklist
- Quantified market motivation in opening paragraphs.
- Explicit mechanism channel statements before regressions.
- Equation or notation-ready variable definitions for core constructs.
- Distinction between correlation evidence and causal claims.
- Clear acknowledgment of contradictory or puzzling coefficients.
- Identification plan with relevance and exclusion discussion.
- Figure/table references and data-source notes where applicable.

## Resources
- `references/reference_style.md`: Full text-extracted version (all pages) — use only if you need the entire document.
- `references/chapter_0_cover_intro.md`: Cover + Introduction + Research Question (pages 1–5).
- `references/chapter_1_literature.md`: Literature review sections (pages 6–7).
- `references/chapter_2_data.md`: Data sources and indexing filter (pages 8–9).
- `references/chapter_3_analysis.md`: Preliminary analysis and regression setup (pages 10–12).
- `references/chapter_4_future_work.md`: Future work and identification strategy (pages 13–14).
- `references/chapter_5_references.md`: Bibliography (pages 15–18).
