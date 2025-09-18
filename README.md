### README — AI Frontier Compute HHI (Rolling 4Q)

#### One-line value statement
Measure and monitor market concentration of frontier AI compute with a reproducible HHI that blends hardware capacity (TFLOPs) and inference throughput (tokens) into a single, auditable metric.

#### TL;DR
In the global frontier AI compute market (>100B params), we calculated a rolling 4-quarter HHI using a composite share: 40% hardware (TFLOPs) + 60% inference (tokens). The HHI remains in high concentration for Q3 2024–Q2 2025, with triggers to audit access practices. Includes methodology, uncertainty bands, specs for BI, and trigger rules for governance.

#### Problem & Context
- Why it matters: Concentration in compute (chips, DCs, APIs, power) shifts economic power faster than GDP-based policy responds. We need an operational, auditable metric to inform governance and procurement.
- Success criteria: Reproducible HHI with clear assumptions, traceable evidence, sensitivity bands, and automatic trigger rules aligned to policy actions.

#### My Role & Scope
- Role: Metric Architect / Data Analyst
- Timeline: 4–6 weeks (initial), then quarterly refresh
- Stakeholders: Policy teams, Competition authorities, CIO/CTO offices, Research/Academia
- Responsibilities: Market definition, methodology, data triangulation, code, uncertainty analysis, reporting, and governance triggers

#### Data Overview
- Sources: SEC filings, cloud provider docs (rate limits/pricing), industry research; triangulated where enterprise usage is opaque
- Period: Q3 2024–Q2 2025 (rolling 4Q); monthly when feasible, otherwise quarterly
- Dimensions: Hardware (accelerators TFLOPs); Inference (tokens processed)
- Data quality: Hardware high; Inference medium (triangulation). Evidence package includes hashes and access dates.

#### Questions & Hypotheses
- Q1: Is compute access concentrated enough to warrant governance triggers?
- Q2: Do access constraints (tiers/ToS) persist despite capacity increases?
- Hypothesis: HHI stays >2,500 across rolling 4Q; triggers for access audit are warranted.

#### Approach & Methods
- Pipeline: ingest → normalize → entity mapping → composite share → HHI → sensitivity → triggers → deliverables
- Methods: FLOPs normalization, token estimation via triangulation, vertical-integration mapping, ±60% sensitivity
- Rationale: Composite share captures the “who can run models” reality (execution) and the physical substrate (chips).

#### Key Formulas


#### Key Insights
1) HHI remains ~2,845–2,857 across Q3 2024–Q2 2025 → High concentration (≥2,500).
2) Azure/OpenAI ecosystem and NVIDIA/OpenAI dominate final shares; Google (TPU+Gemini) and Anthropic+AWS follow.
3) Sensitivity (±60%) preserves “high” band; evidence confidence medium (inference triangulation).

#### Impact & Outcomes
- Decisions enabled: Trigger “audit_access_practices,” review anti-benchmark clauses, plan for interoperability in civic APIs if HHI escalates.
- KPI movement: Persistent >2,500 HHI (4Q) activates governance workflow.
- ROI: Clear policy playbook reduces audit time and improves procurement leverage.

#### Evaluation & Metrics
- Concentration bands: <1,500 healthy; 1,500–2,500 moderate; 2,500–5,000 high; ≥5,000 extreme
- Triggers: 4Q >2,500 (audit); 2Q >3,000 (ban anti-benchmark in public contracts); any >4,000 (mandatory interoperability)
- Uncertainty: ±60% scenarios; report min/median/max

#### Constraints & Trade-offs
- Sparse monthly inference data; triangulation required
- Latency vs. interpretability: kept method simple, documented assumptions
- Vendor opacity: mitigated via evidence package and sensitivity

#### Ethics & Privacy
- Respect ToS/licenses; avoid paywalled proprietary data beyond citables
- Publish hashes, access dates, and snapshots where permitted

#### Lessons Learned
- Final shares are highly sensitive to inference estimates; transparent bands are essential
- Entity mapping drives interpretability; keep rationale explicit

#### Tech Stack
- Python (pandas, numpy, plotly+kaleido), JSON

#### Reproducibility
- Data access: Provided JSONs under data/raw and processed results under data/processed
- Environment: see requirements (below)
- Run:
  - Update file paths in src/ai_frontier_hhi_calculator.py main() to use repo-relative paths (see Quickstart)
  - Run: python src/ai_frontier_hhi_calculator.py
- Artifacts produced: chart PNG, executive report MD, specs JSON, complete results JSON


#### Contact / CTA
- Open an issue for questions, or reach out via the repo’s Discussions tab.
