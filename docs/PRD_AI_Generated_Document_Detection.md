# Product Requirements Document: AI-Generated Document Detection System

**Document Version:** 1.0
**Date:** December 14, 2025
**Status:** Draft for Review
**Classification:** Internal - Confidential

---

## Executive Summary

FinTechCo has experienced a **338% surge in Synthetic ID fraud** (June-November 2025), resulting in $420K in losses YTD 2025. Current ID validation systems detect traditional forgery (photo swaps, tampered text, fake holograms) but fail to identify fully AI-generated documents. This PRD defines requirements for an enhanced detection system to block AI-generated synthetic IDs.

**Business Impact:**
- **Current Loss Rate:** $420K YTD 2025 (152% increase YoY)
- **Target Reduction:** 50% reduction in synthetic ID fraud losses within 6 months
- **Strategic Priority:** Critical - addresses largest fraud growth vector

---

## Problem Statement

### Current State
The existing vision-based ID validation model (`scripts/id_card_validation_model.py`) achieves 100% accuracy on traditional forgery but has a critical blind spot:

| Detection Capability | Status | Performance |
|---------------------|--------|-------------|
| Photo swap detection | ✅ Working | 100% accuracy |
| Fake hologram detection | ✅ Working | 100% accuracy |
| Altered text detection | ✅ Working | 100% accuracy |
| **AI-generated document detection** | ❌ **Missing** | **0% detection** |

### Problem Definition
AI-generated documents bypass traditional forgery checks because they:
1. Have no physical tampering artifacts (no lighting inconsistencies from photo swaps)
2. Generate all elements digitally (no font/alignment anomalies)
3. Create realistic security features (holograms, microprint appear authentic)
4. Pass template matching (trained on authentic document images)

### Impact Analysis
- **Financial:** $420K losses YTD, trending to $650K+ annually
- **Regulatory:** KYC/AML compliance risk if fraudulent IDs pass validation
- **Reputational:** Account opening fraud undermines customer trust
- **Operational:** Manual review queue overload (current: unknown, target: <15%)

---

## Goals & Success Metrics

### Primary Goals

#### Goal 1: Detect AI-Generated Documents
**Objective:** Identify documents created by generative AI tools (Stable Diffusion, Midjourney, DALL-E, etc.)

**Success Metrics:**
| Metric | Baseline | Target (6 months) | Measurement |
|--------|----------|-------------------|-------------|
| AI Detection Rate (TPR) | 0% | ≥85% | True positives / (TP + FN) |
| False Positive Rate | <1% | <3% | Legitimate IDs flagged as AI |
| Synthetic ID Fraud Losses | $420K YTD | <$200K (next 6 mo) | Financial impact |
| Detection Latency (p99) | N/A | <500ms | End-to-end validation time |

#### Goal 2: Maintain Legitimate User Experience
**Objective:** Minimize friction for real customers submitting authentic IDs

**Success Metrics:**
| Metric | Target | Measurement |
|--------|--------|-------------|
| False Positive Rate | <3% | Legitimate IDs incorrectly rejected |
| Manual Review Rate | <15% | % of submissions requiring human review |
| User Abandonment Rate | No increase | Drop-off during ID submission |

#### Goal 3: Operational Scalability
**Objective:** Handle production volume without infrastructure bottlenecks

**Success Metrics:**
| Metric | Target | Measurement |
|--------|--------|-------------|
| Throughput | 1000 validations/min | Peak capacity |
| Latency p50 | <200ms | Median response time |
| Latency p99 | <500ms | 99th percentile |

### Secondary Goals
- **Explainability:** Provide fraud analysts with visual indicators of AI artifacts
- **Continuous Learning:** Active learning pipeline to adapt to new AI generation techniques
- **Multi-Format Support:** Detect AI generation across all ID types (drivers licenses, passports, national IDs)

---

## User Stories & Use Cases

### User Story 1: Fraud Analyst Reviews Suspicious ID
**As a** fraud analyst
**I want to** see visual indicators of AI generation artifacts
**So that** I can make informed decisions on manual review cases

**Acceptance Criteria:**
- System highlights suspicious image regions (frequency anomalies, unnatural edges)
- Fraud probability score includes breakdown by detection method (AI vs. traditional forgery)
- Analyst can see comparison of submitted ID vs. authentic template

**Priority:** High

---

### User Story 2: Customer Submits Authentic ID
**As a** legitimate customer
**I want to** submit my driver's license for account opening
**So that** I can start using FinTechCo services quickly

**Acceptance Criteria:**
- Authentic ID approved within 2 seconds (95% of cases)
- No false rejection of legitimate documents
- Clear feedback if ID quality is insufficient (blur, glare, etc.)

**Priority:** Critical

---

### User Story 3: Fraudster Attempts AI-Generated ID Submission
**As a** fraud prevention system
**I want to** detect AI-generated IDs at submission time
**So that** fraudulent accounts are blocked before creation

**Acceptance Criteria:**
- AI-generated IDs flagged with ≥85% accuracy
- Fraudulent submission blocked immediately (no account creation)
- Fraud case logged for analysis with full metadata

**Priority:** Critical

---

### User Story 4: Data Scientist Monitors Model Performance
**As a** data scientist
**I want to** track detection performance metrics in real-time
**So that** I can identify model drift and trigger retraining

**Acceptance Criteria:**
- Dashboard shows TPR, FPR, latency, and drift metrics
- Alerting when metrics fall below thresholds
- Access to labeled test set for validation

**Priority:** High

---

### User Story 5: System Handles Novel AI Generation Technique
**As the** AI detection system
**I want to** adapt to new AI generation methods automatically
**So that** detection remains effective as adversaries evolve

**Acceptance Criteria:**
- Active learning pipeline flags uncertain cases (0.4-0.6 probability)
- Manual review labels feed back into retraining pipeline
- Weekly model updates without manual intervention

**Priority:** Medium

---

## Functional Requirements

### FR-1: AI Artifact Detection
**Priority:** P0 (Critical)

**Description:** Detect AI generation artifacts in frequency domain, noise patterns, and compression signatures

**Requirements:**
- FR-1.1: Perform FFT (Fast Fourier Transform) analysis to detect spectral anomalies
- FR-1.2: Analyze noise distribution for non-sensor patterns (Gaussian vs. PRNU)
- FR-1.3: Detect JPEG compression artifacts characteristic of AI generation
- FR-1.4: Identify checkerboard artifacts from GAN upsampling
- FR-1.5: Validate Photo Response Non-Uniformity (camera fingerprint)

**Acceptance Criteria:**
- Each feature generates 0-1 confidence score
- Ensemble of FR-1.1-1.5 achieves ≥70% detection rate independently

---

### FR-2: Semantic Consistency Validation
**Priority:** P0 (Critical)

**Description:** Detect logical impossibilities in AI-generated documents

**Requirements:**
- FR-2.1: Cross-validate issue date vs. document security features timeline
- FR-2.2: Verify geographic consistency (state/country vs. template version)
- FR-2.3: Check regulatory consistency (features match issuance period)
- FR-2.4: Validate text-image consistency (OCR data matches visual)
- FR-2.5: Detect missing digital watermarks (present in real IDs post-2020)

**Acceptance Criteria:**
- Each validation generates pass/fail + confidence score
- Single failed consistency check elevates risk score by ≥20%

---

### FR-3: Deepfake Face Detection
**Priority:** P0 (Critical)

**Description:** Identify AI-generated or manipulated facial images

**Requirements:**
- FR-3.1: Analyze eye reflection consistency (lighting physics)
- FR-3.2: Validate pupil light response characteristics
- FR-3.3: Detect unnatural facial landmark distributions
- FR-3.4: Score skin texture realism (pore patterns, wrinkles)
- FR-3.5: Estimate 3D geometry consistency across features

**Acceptance Criteria:**
- Deepfake detector achieves ≥80% accuracy on FaceForensics++ benchmark
- Integrates with existing face detection pipeline (scripts/id_card_validation_model.py:113-129)

---

### FR-4: Two-Stage Detection Architecture
**Priority:** P0 (Critical)

**Description:** Implement fast screening + deep analysis pipeline

**Requirements:**
- FR-4.1: Stage 1 runs traditional forgery checks (<100ms latency)
- FR-4.2: Stage 2 triggers on high-risk cases or random sample (10%)
- FR-4.3: Stage 2 performs AI detection with <500ms latency
- FR-4.4: Risk scores from both stages combine via weighted ensemble

**Acceptance Criteria:**
- 90% of legitimate IDs approved in Stage 1 (<100ms)
- 100% of high-risk IDs proceed to Stage 2
- Combined pipeline meets <500ms p99 latency SLA

---

### FR-5: Risk Score Fusion
**Priority:** P0 (Critical)

**Description:** Combine traditional forgery, AI detection, and deepfake scores

**Requirements:**
- FR-5.1: Traditional forgery weight = 50% if AI score <0.7, else 10%
- FR-5.2: AI detection weight = 60% if AI score ≥0.7, else 30%
- FR-5.3: Deepfake weight = 30% if face detected, else 0%
- FR-5.4: Final score maps to risk categories: Low (<0.3), Medium (0.3-0.6), High (0.6-0.8), Critical (≥0.8)

**Acceptance Criteria:**
- Risk score calculation documented and auditable
- Threshold tuning interface for fraud analysts
- A/B testing framework to optimize weights

---

### FR-6: Explainability & Visualization
**Priority:** P1 (High)

**Description:** Provide fraud analysts with interpretable results

**Requirements:**
- FR-6.1: Highlight image regions triggering AI detection (heatmap)
- FR-6.2: Display feature importance breakdown (which check failed)
- FR-6.3: Compare submitted ID to authentic template side-by-side
- FR-6.4: Generate PDF report for regulatory compliance

**Acceptance Criteria:**
- Fraud analyst can explain rejection reason in <30 seconds
- Heatmap visualizations match model attention (Grad-CAM)
- PDF reports include all metadata for audit trail

---

### FR-7: Active Learning Pipeline
**Priority:** P1 (High)

**Description:** Continuously improve detection via manual review feedback

**Requirements:**
- FR-7.1: Flag uncertain predictions (0.4-0.6 probability) for manual review
- FR-7.2: Fraud analyst labels reviewed cases (legitimate / fraudulent / unsure)
- FR-7.3: Labeled data feeds into weekly retraining pipeline
- FR-7.4: Model versioning with rollback capability

**Acceptance Criteria:**
- ≥500 labeled samples per week from production traffic
- Retraining improves test set accuracy by ≥2% per iteration
- Zero-downtime model deployment

---

### FR-8: Multi-Format Support
**Priority:** P2 (Medium)

**Description:** Detect AI generation across all ID types

**Requirements:**
- FR-8.1: Support drivers licenses (US 50 states + territories)
- FR-8.2: Support passports (US + 50 international countries)
- FR-8.3: Support national IDs (10+ countries)
- FR-8.4: Support state IDs and military IDs

**Acceptance Criteria:**
- Detection accuracy ≥85% for each ID type
- Template library covers 90% of submitted ID types

---

## Non-Functional Requirements

### NFR-1: Performance
- **Latency:** p99 <500ms end-to-end validation
- **Throughput:** 1000 validations/minute sustained
- **Availability:** 99.9% uptime (43 minutes downtime/month)

### NFR-2: Scalability
- **Horizontal Scaling:** Auto-scale to 10x peak load
- **GPU Utilization:** ≥70% average utilization
- **Storage:** Support 10M ID image archive (1TB)

### NFR-3: Security
- **Data Encryption:** At-rest (AES-256) and in-transit (TLS 1.3)
- **PII Protection:** ID images retained 90 days, then deleted
- **Access Control:** Role-based access (fraud analyst, data scientist, admin)
- **Audit Logging:** All predictions and manual reviews logged

### NFR-4: Compliance
- **KYC/AML:** Meets FFIEC guidance on customer identification
- **Data Privacy:** GDPR/CCPA compliant (right to deletion)
- **Model Governance:** Versioned, auditable, explainable

### NFR-5: Monitoring & Observability
- **Metrics:** TPR, FPR, latency, throughput, error rate
- **Alerting:** PagerDuty integration for critical failures
- **Logging:** Structured logs with trace IDs
- **Dashboards:** Real-time Grafana dashboards for fraud team

---

## Out of Scope (V1)

The following capabilities are explicitly **not included** in V1:

1. **Liveness Detection:** Preventing video replay attacks (planned for V2)
2. **Face Matching:** Comparing ID photo to selfie (planned for V2)
3. **Temporal Analysis:** Comparing multiple ID submissions over time (V2)
4. **External Database Integration:** DMV/passport database cross-validation (V3)
5. **Mobile Edge Deployment:** On-device document scanning (V3)
6. **International Multi-Language OCR:** Beyond English/Spanish (V2)

---

## Dependencies & Assumptions

### Dependencies
- **Training Data:** Access to 100K+ authentic IDs (DMV partnership required)
- **GPU Infrastructure:** 4x NVIDIA A100 GPUs for training/inference
- **ML Platform:** Existing Snowflake + Python environment
- **Labeling Resources:** 2 FTE fraud analysts for manual review

### Assumptions
- Synthetic ID fraud trend continues at current rate (338% surge)
- AI generation techniques remain detectable via frequency analysis
- False positive rate <3% is acceptable to business stakeholders
- Current ID submission volume: <100 validations/minute peak

### Risks
| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Adversarial adaptation | High | High | Active learning + red team testing |
| Training data availability | Medium | High | Start with synthetic data + public datasets |
| False positive customer churn | Medium | Medium | Gradual rollout with shadow mode |
| GPU infrastructure cost | Low | Medium | Optimize batch inference, use Haiku for fast path |

---

## Success Criteria & Launch Plan

### Phase 1: Proof of Concept (Weeks 1-4)
**Goal:** Validate AI detection feasibility on existing synthetic ID fraud cases

**Deliverables:**
- Analyze 340 confirmed synthetic fraud cases (June-Nov 2025)
- Implement FR-1 (AI artifact detection) as standalone module
- Achieve ≥70% detection rate on labeled fraud cases

**Success Criteria:**
- Detection rate ≥70% on known fraud cases
- False positive rate <10% on legitimate test set (acceptable for PoC)

---

### Phase 2: Alpha Deployment (Weeks 5-12)
**Goal:** Integrate AI detection into production pipeline in shadow mode

**Deliverables:**
- Implement FR-1, FR-2, FR-3, FR-4, FR-5 (core detection)
- Deploy in shadow mode (no blocking, logging only)
- Collect 10K+ production samples with labels

**Success Criteria:**
- Shadow mode accuracy ≥80% (validated via manual review)
- False positive rate <5%
- p99 latency <750ms (optimization in Beta)

---

### Phase 3: Beta Launch (Weeks 13-20)
**Goal:** Enable blocking mode for high-confidence cases

**Deliverables:**
- Block IDs with AI probability ≥0.9 (critical risk)
- Medium/high risk → manual review queue
- Implement FR-6 (explainability) and FR-7 (active learning)

**Success Criteria:**
- Synthetic ID fraud losses reduced by ≥30%
- False positive rate <3%
- p99 latency <500ms
- Manual review queue <15% of submissions

---

### Phase 4: General Availability (Weeks 21-24)
**Goal:** Full production rollout with continuous monitoring

**Deliverables:**
- Lower blocking threshold to ≥0.8 based on Beta performance
- Enable auto-scaling and load balancing
- Implement NFR-1 through NFR-5 (all non-functional requirements)

**Success Criteria:**
- 50% reduction in synthetic ID fraud losses (vs. Q3/Q4 2025 baseline)
- TPR ≥85%, FPR <3%
- 99.9% uptime
- Active learning pipeline running weekly

---

## Stakeholders & Roles

| Role | Name | Responsibility |
|------|------|----------------|
| **Product Owner** | [TBD] | Prioritization, business requirements |
| **Engineering Lead** | [TBD] | Technical architecture, delivery |
| **Data Science Lead** | [TBD] | Model development, performance |
| **Fraud Operations** | [TBD] | Manual review, false positive analysis |
| **InfoSec** | [TBD] | Security review, PII compliance |
| **Legal/Compliance** | [TBD] | Regulatory requirements, audit |

---

## Appendix A: Competitive Analysis

| Provider | AI Detection Capability | Accuracy | Notes |
|----------|------------------------|----------|-------|
| Persona | ✅ Yes | Unknown | Proprietary black box |
| Onfido | ✅ Yes | ~90% (claimed) | Uses deepfake detection |
| Jumio | ✅ Yes | Unknown | Frequency analysis mentioned |
| Socure | ✅ Yes | Unknown | Multi-modal approach |
| **FinTechCo (Current)** | ❌ No | 0% | Traditional forgery only |
| **FinTechCo (Target)** | ✅ Yes | ≥85% | This PRD |

---

## Appendix B: Related Documents

- [Technical Specification: AI-Generated Document Detection](./TechSpec_AI_Generated_Document_Detection.md)
- [Macro Risk Assessment Dec 2025](../data/docs/Macro_Risk_Assessment_Dec2025_Validation.md)
- [Existing ID Validation Model](../scripts/id_card_validation_model.py)
- [High-Level Implementation Plan](../README.md) - see "Model Improvements & Next Steps"

---

## Approval & Sign-Off

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Product Owner | [TBD] | | |
| Engineering Lead | [TBD] | | |
| Data Science Lead | [TBD] | | |
| Fraud Operations | [TBD] | | |

---

**Document History:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-14 | Claude Code | Initial draft |

---

**Next Steps:**
1. Review PRD with stakeholders
2. Obtain sign-off from Product Owner, Engineering Lead, Data Science Lead
3. Proceed to Technical Specification development
4. Initiate Phase 1: Proof of Concept
