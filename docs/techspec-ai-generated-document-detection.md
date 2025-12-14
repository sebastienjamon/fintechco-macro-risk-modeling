# Technical Specification: AI-Generated Document Detection System

**Document Version:** 1.0
**Date:** December 14, 2025
**Status:** Draft for Engineering Review
**Classification:** Internal - Confidential

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Data Models](#data-models)
4. [Feature Engineering](#feature-engineering)
5. [Machine Learning Models](#machine-learning-models)
6. [API Specifications](#api-specifications)
7. [Infrastructure](#infrastructure)
8. [Security & Compliance](#security--compliance)
9. [Testing Strategy](#testing-strategy)
10. [Deployment Plan](#deployment-plan)
11. [Monitoring & Observability](#monitoring--observability)
12. [Appendices](#appendices)

---

## System Overview

### Purpose
Extend the existing ID card validation system (`scripts/id_card_validation_model.py`) to detect fully AI-generated documents, addressing the 338% surge in synthetic ID fraud.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     ID Validation Service                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Client Upload → Image Processing → Detection Pipeline         │
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐      │
│  │   Stage 1    │   │   Stage 2    │   │   Risk       │      │
│  │  Traditional │──→│  AI Detection│──→│   Scoring    │      │
│  │   Forgery    │   │   + Deepfake │   │   + Action   │      │
│  └──────────────┘   └──────────────┘   └──────────────┘      │
│        ↓                   ↓                   ↓               │
│  <100ms latency      <500ms latency      Decision             │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │        Active Learning & Retraining Pipeline             │ │
│  └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Layer | Technology | Justification |
|-------|------------|---------------|
| **Data Storage** | Snowflake | Existing platform, ACID compliance |
| **ML Training** | Python 3.11, PyTorch 2.1 | GPU support, model ecosystem |
| **ML Inference** | ONNX Runtime, TensorRT | Low-latency optimization |
| **API Gateway** | FastAPI | Async support, auto-docs |
| **Image Processing** | OpenCV 4.8, PIL 10.0 | Standard CV libraries |
| **Feature Store** | Feast | Feature versioning, serving |
| **Model Registry** | MLflow | Experiment tracking, versioning |
| **Monitoring** | Prometheus + Grafana | Time-series metrics |
| **Logging** | ELK Stack | Centralized log aggregation |
| **Compute** | AWS EC2 g5.xlarge (inference) | NVIDIA A10G GPU |
| **Training** | AWS EC2 p4d.24xlarge | 8x NVIDIA A100 (40GB) |

---

## Architecture

### Component Diagram

```
┌───────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                              │
├───────────────────────────────────────────────────────────────────┤
│  Web Upload | Mobile App | Batch API | Manual Review UI          │
└────────────────────────────┬──────────────────────────────────────┘
                             │
                             ↓
┌───────────────────────────────────────────────────────────────────┐
│                     API GATEWAY (FastAPI)                         │
├───────────────────────────────────────────────────────────────────┤
│  • Authentication (JWT)                                           │
│  • Rate limiting (1000/min per client)                            │
│  • Request validation                                             │
│  • Load balancing                                                 │
└────────────────────────────┬──────────────────────────────────────┘
                             │
                             ↓
┌───────────────────────────────────────────────────────────────────┐
│                   IMAGE PREPROCESSING SERVICE                     │
├───────────────────────────────────────────────────────────────────┤
│  • Image quality checks (blur, glare, size)                       │
│  • Normalization (resize to 512x512)                              │
│  • Face detection & cropping                                      │
│  • Metadata extraction (EXIF)                                     │
└────────────────────────────┬──────────────────────────────────────┘
                             │
                             ↓
┌───────────────────────────────────────────────────────────────────┐
│                   DETECTION PIPELINE ORCHESTRATOR                 │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              STAGE 1: TRADITIONAL FORGERY                   │ │
│  ├─────────────────────────────────────────────────────────────┤ │
│  │  Existing Model: Random Forest (21 features)                │ │
│  │  Latency: <100ms                                            │ │
│  │  Decision:                                                  │ │
│  │    • fraud_prob < 0.3 → APPROVE (skip Stage 2)             │ │
│  │    • fraud_prob ≥ 0.3 → STAGE 2                            │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                             ↓                                     │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              STAGE 2: AI DETECTION                          │ │
│  ├─────────────────────────────────────────────────────────────┤ │
│  │  Parallel Execution:                                        │ │
│  │    1. AI Artifact Detector (frequency, noise, PRNU)        │ │
│  │    2. Semantic Validator (temporal, geographic logic)      │ │
│  │    3. Deepfake Detector (face analysis)                    │ │
│  │  Latency: <500ms                                           │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                             ↓                                     │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              RISK SCORE FUSION                              │ │
│  ├─────────────────────────────────────────────────────────────┤ │
│  │  Weighted ensemble of Stage 1 + Stage 2 outputs            │ │
│  │  Final risk: [0.0 - 1.0]                                   │ │
│  │  Categories: Low | Medium | High | Critical                │ │
│  └─────────────────────────────────────────────────────────────┘ │
└────────────────────────────┬──────────────────────────────────────┘
                             │
                             ↓
┌───────────────────────────────────────────────────────────────────┐
│                       DECISION ENGINE                             │
├───────────────────────────────────────────────────────────────────┤
│  • risk < 0.3: AUTO-APPROVE                                       │
│  • 0.3 ≤ risk < 0.8: MANUAL REVIEW                               │
│  • risk ≥ 0.8: AUTO-REJECT                                        │
└────────────────────────────┬──────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              ↓                             ↓
┌──────────────────────┐      ┌──────────────────────────┐
│  SNOWFLAKE STORAGE   │      │  ACTIVE LEARNING QUEUE   │
├──────────────────────┤      ├──────────────────────────┤
│  • Predictions       │      │  • Uncertain cases       │
│  • Features          │      │  • Manual labels         │
│  • Images (90 days)  │      │  • Retraining trigger    │
└──────────────────────┘      └──────────────────────────┘
```

### Data Flow

```
1. Client uploads ID image (JPEG/PNG, <5MB)
2. API Gateway authenticates, rate-limits, forwards to preprocessing
3. Preprocessing validates image quality, extracts metadata
4. Stage 1 (Traditional) runs on all submissions
   - If low risk (< 0.3): APPROVE, log, exit
   - If ≥ 0.3: continue to Stage 2
5. Stage 2 (AI Detection) runs three parallel models:
   - AI Artifact Detector
   - Semantic Validator
   - Deepfake Detector
6. Risk Fusion combines Stage 1 + Stage 2 scores
7. Decision Engine determines action:
   - Auto-approve, manual review, or auto-reject
8. Response returned to client (<500ms p99)
9. All predictions logged to Snowflake
10. Uncertain cases (0.4-0.6) flagged for active learning
```

---

## Data Models

### Database Schema (Snowflake)

#### Table: `ID_VALIDATION_SUBMISSIONS`

```sql
CREATE TABLE ID_VALIDATION_SUBMISSIONS (
    submission_id VARCHAR(36) PRIMARY KEY,           -- UUID
    customer_id VARCHAR(36) NOT NULL,                -- Customer reference
    submission_timestamp TIMESTAMP_NTZ NOT NULL,

    -- Image metadata
    image_url VARCHAR(512),                          -- S3 path
    image_hash VARCHAR(64),                          -- SHA-256
    image_size_bytes INTEGER,
    image_width INTEGER,
    image_height INTEGER,
    image_format VARCHAR(10),                        -- JPEG, PNG

    -- EXIF metadata
    exif_camera_make VARCHAR(100),
    exif_camera_model VARCHAR(100),
    exif_software VARCHAR(100),
    exif_datetime TIMESTAMP_NTZ,

    -- ID document info
    id_type VARCHAR(50),                             -- drivers_license, passport, etc.
    issuing_country VARCHAR(3),                      -- ISO 3166-1 alpha-3
    issuing_state VARCHAR(50),                       -- For US drivers licenses

    -- Model predictions
    stage1_fraud_probability FLOAT,                  -- Traditional forgery score
    stage2_ai_probability FLOAT,                     -- AI generation score
    stage2_deepfake_probability FLOAT,               -- Face manipulation score
    final_risk_score FLOAT,                          -- Fused score
    risk_category VARCHAR(20),                       -- low_risk, medium_risk, high_risk, critical_risk

    -- Decision
    decision VARCHAR(20),                            -- approved, rejected, manual_review
    decision_reason TEXT,                            -- Human-readable explanation

    -- Manual review (if applicable)
    manual_review_assigned_to VARCHAR(100),
    manual_review_timestamp TIMESTAMP_NTZ,
    manual_label VARCHAR(20),                        -- legitimate, fraudulent, unsure
    manual_notes TEXT,

    -- Metadata
    model_version_stage1 VARCHAR(20),
    model_version_stage2 VARCHAR(20),
    processing_latency_ms INTEGER,

    -- Audit
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    updated_at TIMESTAMP_NTZ
);

CREATE INDEX idx_customer_id ON ID_VALIDATION_SUBMISSIONS(customer_id);
CREATE INDEX idx_submission_timestamp ON ID_VALIDATION_SUBMISSIONS(submission_timestamp);
CREATE INDEX idx_manual_review ON ID_VALIDATION_SUBMISSIONS(decision) WHERE decision = 'manual_review';
CREATE INDEX idx_risk_category ON ID_VALIDATION_SUBMISSIONS(risk_category);
```

#### Table: `ID_VALIDATION_FEATURES`

```sql
CREATE TABLE ID_VALIDATION_FEATURES (
    submission_id VARCHAR(36) PRIMARY KEY,

    -- Stage 1: Traditional forgery features (existing 21 features)
    image_sharpness FLOAT,
    brightness_score FLOAT,
    contrast_score FLOAT,
    hologram_detected BOOLEAN,
    hologram_confidence FLOAT,
    microprint_quality FLOAT,
    uv_features_detected BOOLEAN,
    uv_confidence FLOAT,
    ocr_confidence FLOAT,
    font_consistency_score FLOAT,
    text_alignment_score FLOAT,
    face_detected BOOLEAN,
    face_detection_confidence FLOAT,
    face_quality_score FLOAT,
    face_symmetry_score FLOAT,
    edge_consistency FLOAT,
    lighting_consistency FLOAT,
    color_histogram_score FLOAT,
    template_match_score FLOAT,
    is_expired BOOLEAN,
    document_age_years FLOAT,

    -- Stage 2: AI artifact features
    fft_spectral_anomaly_score FLOAT,               -- Frequency domain analysis
    noise_distribution_score FLOAT,                 -- Gaussian vs. sensor noise
    prnu_confidence FLOAT,                          -- Camera fingerprint
    jpeg_compression_anomaly FLOAT,                 -- Compression artifacts
    checkerboard_artifact_score FLOAT,              -- GAN upsampling
    edge_coherence_score FLOAT,                     -- Unnaturally smooth edges

    -- Stage 2: Semantic consistency features
    temporal_consistency BOOLEAN,                   -- Issue date vs. features
    geographic_consistency BOOLEAN,                 -- State/country vs. template
    regulatory_consistency BOOLEAN,                 -- Features match regulations
    text_image_consistency FLOAT,                   -- OCR vs. visual
    digital_watermark_present BOOLEAN,              -- Post-2020 IDs

    -- Stage 2: Deepfake features
    eye_reflection_consistency FLOAT,
    pupil_light_response_score FLOAT,
    facial_landmark_distribution FLOAT,
    skin_texture_realism FLOAT,
    face_3d_geometry_score FLOAT,

    FOREIGN KEY (submission_id) REFERENCES ID_VALIDATION_SUBMISSIONS(submission_id)
);
```

#### Table: `MODEL_VERSIONS`

```sql
CREATE TABLE MODEL_VERSIONS (
    version_id VARCHAR(36) PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,                -- stage1_traditional, stage2_ai, stage2_deepfake
    version_number VARCHAR(20) NOT NULL,             -- 1.0.0, 1.1.0, etc.

    -- Training metadata
    training_dataset_size INTEGER,
    training_start_timestamp TIMESTAMP_NTZ,
    training_end_timestamp TIMESTAMP_NTZ,
    training_accuracy FLOAT,
    validation_accuracy FLOAT,
    test_accuracy FLOAT,
    test_tpr FLOAT,                                  -- True positive rate
    test_fpr FLOAT,                                  -- False positive rate
    test_auc FLOAT,

    -- Model artifacts
    model_file_path VARCHAR(512),                    -- S3 path to model weights
    model_config JSON,                               -- Hyperparameters

    -- Deployment
    deployed_timestamp TIMESTAMP_NTZ,
    retired_timestamp TIMESTAMP_NTZ,
    is_active BOOLEAN DEFAULT FALSE,

    -- Metadata
    created_by VARCHAR(100),
    notes TEXT,
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

CREATE INDEX idx_model_name ON MODEL_VERSIONS(model_name);
CREATE INDEX idx_is_active ON MODEL_VERSIONS(is_active);
```

---

## Feature Engineering

### Stage 2: AI Artifact Detection Features

#### 1. Frequency Domain Analysis (FFT)

**Implementation:**
```python
import numpy as np
import cv2

def extract_fft_spectral_anomaly(image: np.ndarray) -> float:
    """
    Detect spectral anomalies in frequency domain characteristic of AI generation.

    Real photographs have predictable frequency patterns based on camera sensors.
    AI-generated images often have anomalies in high-frequency components.

    Args:
        image: RGB image as numpy array (H, W, 3)

    Returns:
        anomaly_score: 0.0 (real-like) to 1.0 (AI-like)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Compute 2D FFT
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude_spectrum = np.abs(fft_shift)

    # Analyze high-frequency energy distribution
    h, w = magnitude_spectrum.shape
    center_h, center_w = h // 2, w // 2

    # Define annular regions (low, mid, high frequency)
    low_freq_radius = min(h, w) // 8
    mid_freq_radius = min(h, w) // 4
    high_freq_radius = min(h, w) // 2

    # Create masks for each region
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center_w)**2 + (y - center_h)**2)

    low_mask = dist_from_center < low_freq_radius
    mid_mask = (dist_from_center >= low_freq_radius) & (dist_from_center < mid_freq_radius)
    high_mask = (dist_from_center >= mid_freq_radius) & (dist_from_center < high_freq_radius)

    # Compute energy in each region
    low_energy = np.sum(magnitude_spectrum[low_mask])
    mid_energy = np.sum(magnitude_spectrum[mid_mask])
    high_energy = np.sum(magnitude_spectrum[high_mask])

    total_energy = low_energy + mid_energy + high_energy
    high_freq_ratio = high_energy / total_energy

    # AI images typically have abnormal high-frequency energy
    # Real photos: high_freq_ratio typically 0.15-0.30
    # AI images: often < 0.10 or > 0.40

    if high_freq_ratio < 0.10:
        anomaly_score = (0.10 - high_freq_ratio) / 0.10  # 0-1 range
    elif high_freq_ratio > 0.40:
        anomaly_score = min((high_freq_ratio - 0.40) / 0.40, 1.0)
    else:
        anomaly_score = 0.0

    return float(np.clip(anomaly_score, 0.0, 1.0))
```

#### 2. PRNU (Photo Response Non-Uniformity) Analysis

**Implementation:**
```python
def extract_prnu_confidence(image: np.ndarray) -> float:
    """
    Detect camera sensor fingerprint (PRNU pattern).

    Real cameras have unique sensor noise patterns. AI-generated images lack this.

    Args:
        image: RGB image as numpy array

    Returns:
        confidence: 0.0 (no PRNU, likely AI) to 1.0 (strong PRNU, likely real)
    """
    from scipy.ndimage import gaussian_filter

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Denoising filter to estimate noiseless image
    denoised = gaussian_filter(gray, sigma=2)

    # Extract noise residual (PRNU + random noise)
    noise_residual = gray - denoised

    # Compute local variance of noise
    noise_variance = np.var(noise_residual)

    # Real cameras: noise variance typically 10-100
    # AI images: noise variance typically < 5 (too clean) or > 200 (unrealistic)

    if 10 <= noise_variance <= 100:
        confidence = 1.0  # Strong PRNU signature
    elif noise_variance < 5:
        confidence = 0.0  # Too clean, likely AI
    else:
        confidence = max(0.0, 1.0 - (noise_variance - 100) / 100)

    return float(np.clip(confidence, 0.0, 1.0))
```

#### 3. JPEG Compression Artifact Detection

**Implementation:**
```python
def extract_jpeg_compression_anomaly(image: np.ndarray) -> float:
    """
    Detect anomalous JPEG compression patterns.

    Real photos: Single compression from camera
    AI-generated: Often double-compressed or lack compression artifacts

    Args:
        image: RGB image

    Returns:
        anomaly_score: 0.0 (normal) to 1.0 (anomalous)
    """
    # Convert to YCbCr (JPEG color space)
    ycbcr = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    y_channel = ycbcr[:, :, 0]

    # Analyze 8x8 DCT block boundaries (JPEG compression grid)
    h, w = y_channel.shape
    block_size = 8

    # Compute gradient at block boundaries
    vertical_gradients = []
    horizontal_gradients = []

    for i in range(block_size, h, block_size):
        # Vertical block boundaries
        grad = np.abs(y_channel[i, :] - y_channel[i-1, :])
        vertical_gradients.append(np.mean(grad))

    for j in range(block_size, w, block_size):
        # Horizontal block boundaries
        grad = np.abs(y_channel[:, j] - y_channel[:, j-1])
        horizontal_gradients.append(np.mean(grad))

    block_boundary_strength = np.mean(vertical_gradients + horizontal_gradients)

    # Real photos: moderate block boundary artifacts (5-20)
    # AI images: either too weak (<2) or too strong (>30) from double compression

    if 5 <= block_boundary_strength <= 20:
        anomaly_score = 0.0
    elif block_boundary_strength < 2:
        anomaly_score = (2 - block_boundary_strength) / 2
    else:
        anomaly_score = min((block_boundary_strength - 20) / 20, 1.0)

    return float(np.clip(anomaly_score, 0.0, 1.0))
```

#### 4. Checkerboard Artifact Detection (GAN Upsampling)

**Implementation:**
```python
def extract_checkerboard_artifact_score(image: np.ndarray) -> float:
    """
    Detect checkerboard artifacts from GAN upsampling layers.

    GANs using transposed convolutions create checkerboard patterns in high frequencies.

    Args:
        image: RGB image

    Returns:
        artifact_score: 0.0 (no artifact) to 1.0 (strong artifact)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Compute FFT
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)

    h, w = magnitude.shape
    center_h, center_w = h // 2, w // 2

    # Checkerboard pattern appears as peaks at specific frequencies
    # Look for periodicity at 2-pixel intervals (Nyquist frequency)

    # Sample points at checkerboard frequency
    checkerboard_freqs = [
        (center_h + h // 4, center_w),
        (center_h - h // 4, center_w),
        (center_h, center_w + w // 4),
        (center_h, center_w - w // 4),
    ]

    checkerboard_energy = sum(magnitude[y, x] for y, x in checkerboard_freqs)
    total_high_freq_energy = np.sum(magnitude[h//4:3*h//4, w//4:3*w//4])

    # Normalize
    if total_high_freq_energy > 0:
        artifact_score = checkerboard_energy / total_high_freq_energy
    else:
        artifact_score = 0.0

    # Real photos: ratio typically < 0.01
    # GAN images: ratio often > 0.05
    artifact_score = (artifact_score - 0.01) / 0.04  # Scale to 0-1

    return float(np.clip(artifact_score, 0.0, 1.0))
```

### Stage 2: Semantic Consistency Features

#### 5. Temporal Consistency Check

**Implementation:**
```python
from datetime import datetime

def validate_temporal_consistency(
    issue_date: datetime,
    id_type: str,
    security_features: dict
) -> bool:
    """
    Validate that document security features match issuance timeline.

    Example: Real ID Act compliant features only present on US licenses issued after 2020.

    Args:
        issue_date: Document issue date
        id_type: Type of document
        security_features: Dict of detected features (hologram_type, uv_features, etc.)

    Returns:
        consistent: True if features match timeline, False otherwise
    """
    # Example rules (simplified)
    if id_type == "drivers_license" and security_features.get("issuing_country") == "USA":
        # Real ID star logo required for issuance after Jan 2018
        if issue_date >= datetime(2018, 1, 1):
            if not security_features.get("real_id_compliant"):
                return False  # Inconsistent: should have Real ID marker

        # New hologram type introduced in 2020
        if issue_date >= datetime(2020, 1, 1):
            if security_features.get("hologram_type") not in ["type_3", "type_4"]:
                return False  # Inconsistent: should have newer hologram

    # Passport rules
    if id_type == "passport":
        # Biometric passports (with chip) required after 2007
        if issue_date >= datetime(2007, 1, 1):
            if not security_features.get("rfid_chip_detected"):
                return False

    return True
```

#### 6. Geographic Consistency Check

**Implementation:**
```python
def validate_geographic_consistency(
    issuing_state: str,
    issuing_country: str,
    template_version: str
) -> bool:
    """
    Validate that document template matches issuing jurisdiction.

    Example: California driver's license templates changed in 2018, 2020, 2023.

    Args:
        issuing_state: US state (e.g., "CA", "NY")
        issuing_country: Country code (e.g., "USA")
        template_version: Detected template from vision model

    Returns:
        consistent: True if template matches jurisdiction
    """
    # Load template database (simplified)
    TEMPLATE_DB = {
        ("USA", "CA"): ["CA_2018", "CA_2020", "CA_2023"],
        ("USA", "NY"): ["NY_2017", "NY_2021"],
        # ... more states
    }

    valid_templates = TEMPLATE_DB.get((issuing_country, issuing_state), [])

    if template_version not in valid_templates:
        return False  # Template mismatch

    return True
```

### Stage 2: Deepfake Features

#### 7. Eye Reflection Consistency

**Implementation:**
```python
def extract_eye_reflection_consistency(face_crop: np.ndarray) -> float:
    """
    Analyze eye reflections for lighting consistency.

    Real photos: Both eyes reflect light source consistently
    AI-generated: Often inconsistent or missing reflections

    Args:
        face_crop: Cropped face region

    Returns:
        consistency_score: 0.0 (inconsistent) to 1.0 (consistent)
    """
    import dlib

    # Detect facial landmarks (68-point model)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        return 0.5  # Neutral score if no face detected

    shape = predictor(gray, faces[0])

    # Extract eye regions
    left_eye_points = [(shape.part(i).x, shape.part(i).y) for i in range(36, 42)]
    right_eye_points = [(shape.part(i).x, shape.part(i).y) for i in range(42, 48)]

    # Get bounding boxes for eyes
    left_eye_bbox = cv2.boundingRect(np.array(left_eye_points))
    right_eye_bbox = cv2.boundingRect(np.array(right_eye_points))

    left_eye_crop = face_crop[left_eye_bbox[1]:left_eye_bbox[1]+left_eye_bbox[3],
                               left_eye_bbox[0]:left_eye_bbox[0]+left_eye_bbox[2]]
    right_eye_crop = face_crop[right_eye_bbox[1]:right_eye_bbox[1]+right_eye_bbox[3],
                                right_eye_bbox[0]:right_eye_bbox[0]+right_eye_bbox[2]]

    # Detect bright spots (reflections) in each eye
    left_reflections = detect_bright_spots(left_eye_crop)
    right_reflections = detect_bright_spots(right_eye_crop)

    # Check if both eyes have similar reflection patterns
    if len(left_reflections) == 0 or len(right_reflections) == 0:
        consistency_score = 0.3  # Missing reflections
    elif len(left_reflections) != len(right_reflections):
        consistency_score = 0.5  # Different number of reflections
    else:
        # Compare reflection positions (should be symmetric)
        position_similarity = compute_reflection_symmetry(left_reflections, right_reflections)
        consistency_score = position_similarity

    return float(np.clip(consistency_score, 0.0, 1.0))

def detect_bright_spots(eye_crop: np.ndarray, threshold: int = 200) -> list:
    """Detect bright spots (reflections) in eye region."""
    gray = cv2.cvtColor(eye_crop, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours
```

---

## Machine Learning Models

### Model 1: AI Artifact Detector

**Architecture:**
```
Input: 512x512 RGB image
    ↓
EfficientNet-B3 Backbone (pretrained on ImageNet)
    ↓
Feature Extraction (1536-dim)
    ↓
Parallel Branches:
    ├─ Frequency Analysis Branch (FFT features)
    ├─ Noise Analysis Branch (PRNU features)
    └─ Compression Branch (JPEG artifacts)
    ↓
Concatenate (1536 + 64 + 64 + 64 = 1728-dim)
    ↓
Fully Connected Layers:
    ├─ Dense(512, ReLU, Dropout 0.3)
    ├─ Dense(256, ReLU, Dropout 0.3)
    └─ Dense(1, Sigmoid)
    ↓
Output: AI generation probability [0, 1]
```

**Training Configuration:**
```yaml
model_name: ai_artifact_detector
version: 1.0.0

architecture:
  backbone: efficientnet-b3
  input_size: [512, 512, 3]
  output_classes: 1  # Binary classification

training:
  optimizer: AdamW
  learning_rate: 0.0001
  weight_decay: 0.01
  batch_size: 32
  epochs: 100
  early_stopping_patience: 10

  loss_function: BCEWithLogitsLoss
  class_weights: [1.0, 2.0]  # Upweight AI-generated class

  augmentation:
    - horizontal_flip: 0.5
    - rotation: [-10, 10]
    - color_jitter: [0.1, 0.1, 0.1, 0.05]
    - gaussian_blur: 0.3
    - jpeg_compression: [50, 95]

dataset:
  train_size: 70000
  val_size: 15000
  test_size: 15000

  sources:
    - real_ids: 50000  # DMV partnership
    - ai_generated_stable_diffusion: 20000
    - ai_generated_midjourney: 15000
    - ai_generated_dalle3: 15000

evaluation:
  metrics:
    - accuracy
    - precision
    - recall
    - f1_score
    - roc_auc
    - pr_auc

  target_performance:
    test_accuracy: 0.85
    test_recall: 0.85  # True positive rate
    test_precision: 0.90
```

### Model 2: Deepfake Face Detector

**Architecture:**
```
Input: 224x224 RGB face crop
    ↓
XceptionNet Backbone (pretrained on FaceForensics++)
    ↓
Attention Module (focus on eyes, mouth, face boundaries)
    ↓
Feature Extraction (2048-dim)
    ↓
Custom Deepfake Features:
    ├─ Eye reflection analysis
    ├─ Facial landmark consistency
    └─ Skin texture features
    ↓
Concatenate (2048 + 128 = 2176-dim)
    ↓
Fully Connected Layers:
    ├─ Dense(512, ReLU, Dropout 0.4)
    ├─ Dense(256, ReLU, Dropout 0.4)
    └─ Dense(1, Sigmoid)
    ↓
Output: Deepfake probability [0, 1]
```

**Training Configuration:**
```yaml
model_name: deepfake_face_detector
version: 1.0.0

architecture:
  backbone: xception
  input_size: [224, 224, 3]
  pretrained_on: faceforensics++
  attention_mechanism: spatial_attention

training:
  optimizer: Adam
  learning_rate: 0.00005
  batch_size: 64
  epochs: 50

  loss_function: FocalLoss  # Handle class imbalance
  focal_alpha: 0.25
  focal_gamma: 2.0

dataset:
  train_size: 40000
  val_size: 8000
  test_size: 8000

  sources:
    - real_face_crops: 28000
    - deepfake_faceswap: 8000
    - deepfake_face2face: 8000
    - ai_generated_faces: 12000

evaluation:
  target_performance:
    test_accuracy: 0.80
    test_recall: 0.80
    test_auc: 0.85
```

### Risk Fusion Model

**Implementation:**
```python
def compute_final_risk_score(
    stage1_fraud_prob: float,
    stage2_ai_prob: float,
    stage2_deepfake_prob: float,
    face_detected: bool
) -> dict:
    """
    Combine Stage 1 and Stage 2 outputs into final risk score.

    Weighting strategy:
    - If AI probability is high (≥0.7), prioritize Stage 2
    - If AI probability is low (<0.7), balance Stage 1 and Stage 2
    - Deepfake score only matters if face is detected

    Args:
        stage1_fraud_prob: Traditional forgery probability [0, 1]
        stage2_ai_prob: AI generation probability [0, 1]
        stage2_deepfake_prob: Deepfake probability [0, 1]
        face_detected: Whether face was detected

    Returns:
        dict: {
            'final_risk_score': float,
            'risk_category': str,
            'weights': dict
        }
    """
    # Dynamic weighting based on AI probability
    if stage2_ai_prob >= 0.7:
        # High AI signal: prioritize AI detection
        w_stage1 = 0.10
        w_ai = 0.60
        w_deepfake = 0.30 if face_detected else 0.0
    else:
        # Low/medium AI signal: balance both stages
        w_stage1 = 0.50
        w_ai = 0.30
        w_deepfake = 0.20 if face_detected else 0.0

    # Normalize weights if no face detected
    if not face_detected:
        total = w_stage1 + w_ai
        w_stage1 = w_stage1 / total
        w_ai = w_ai / total
        w_deepfake = 0.0

    # Compute weighted average
    final_risk_score = (
        w_stage1 * stage1_fraud_prob +
        w_ai * stage2_ai_prob +
        w_deepfake * stage2_deepfake_prob
    )

    # Clamp to [0, 1]
    final_risk_score = np.clip(final_risk_score, 0.0, 1.0)

    # Categorize risk
    if final_risk_score < 0.3:
        risk_category = "low_risk"
    elif final_risk_score < 0.6:
        risk_category = "medium_risk"
    elif final_risk_score < 0.8:
        risk_category = "high_risk"
    else:
        risk_category = "critical_risk"

    return {
        'final_risk_score': float(final_risk_score),
        'risk_category': risk_category,
        'weights': {
            'stage1_traditional': w_stage1,
            'stage2_ai': w_ai,
            'stage2_deepfake': w_deepfake
        }
    }
```

---

## API Specifications

### REST API Endpoints

#### 1. Submit ID for Validation

**Endpoint:** `POST /api/v1/id-validation/submit`

**Request:**
```json
{
  "customer_id": "cust_abc123",
  "id_type": "drivers_license",
  "issuing_country": "USA",
  "issuing_state": "CA",
  "image": "<base64_encoded_image>",
  "metadata": {
    "submission_source": "web_upload",
    "user_agent": "Mozilla/5.0..."
  }
}
```

**Response (Success):**
```json
{
  "submission_id": "sub_xyz789",
  "decision": "approved",
  "risk_score": 0.15,
  "risk_category": "low_risk",
  "processing_latency_ms": 87,
  "timestamp": "2025-12-14T10:30:00Z"
}
```

**Response (Manual Review):**
```json
{
  "submission_id": "sub_xyz790",
  "decision": "manual_review",
  "risk_score": 0.65,
  "risk_category": "high_risk",
  "reason": "AI generation probability: 0.72",
  "processing_latency_ms": 453,
  "timestamp": "2025-12-14T10:31:00Z"
}
```

**Response (Rejected):**
```json
{
  "submission_id": "sub_xyz791",
  "decision": "rejected",
  "risk_score": 0.93,
  "risk_category": "critical_risk",
  "reason": "High confidence AI-generated document detected",
  "processing_latency_ms": 421,
  "timestamp": "2025-12-14T10:32:00Z"
}
```

#### 2. Get Validation Status

**Endpoint:** `GET /api/v1/id-validation/{submission_id}`

**Response:**
```json
{
  "submission_id": "sub_xyz789",
  "status": "completed",
  "decision": "approved",
  "risk_score": 0.15,
  "detailed_scores": {
    "stage1_traditional_forgery": 0.12,
    "stage2_ai_generation": 0.18,
    "stage2_deepfake": 0.10
  },
  "submission_timestamp": "2025-12-14T10:30:00Z",
  "completion_timestamp": "2025-12-14T10:30:00.087Z"
}
```

#### 3. Manual Review API (Fraud Analyst)

**Endpoint:** `POST /api/v1/id-validation/{submission_id}/review`

**Request:**
```json
{
  "analyst_id": "analyst_john",
  "label": "fraudulent",
  "notes": "Clear AI-generated hologram, unnatural skin texture",
  "fraud_type": "ai_generated_document"
}
```

**Response:**
```json
{
  "submission_id": "sub_xyz790",
  "review_recorded": true,
  "label_applied": "fraudulent",
  "flagged_for_retraining": true,
  "timestamp": "2025-12-14T11:00:00Z"
}
```

#### 4. Batch Validation API

**Endpoint:** `POST /api/v1/id-validation/batch`

**Request:**
```json
{
  "submissions": [
    {
      "customer_id": "cust_001",
      "id_type": "passport",
      "image": "<base64>"
    },
    {
      "customer_id": "cust_002",
      "id_type": "drivers_license",
      "image": "<base64>"
    }
  ]
}
```

**Response:**
```json
{
  "batch_id": "batch_123",
  "total_submissions": 2,
  "status": "processing",
  "estimated_completion_time": "2025-12-14T10:35:00Z"
}
```

---

## Infrastructure

### Compute Resources

#### Development Environment
```yaml
environment: development
region: us-west-2

instances:
  - type: g5.xlarge
    count: 2
    gpu: NVIDIA A10G (24GB)
    vcpu: 4
    ram: 16GB
    purpose: Model development, testing

storage:
  - type: S3
    bucket: fintechco-dev-id-validation
    size: 100GB
    purpose: Images, model artifacts
```

#### Production Environment
```yaml
environment: production
region: us-west-2 (primary), us-east-1 (failover)

api_gateway:
  - type: Application Load Balancer
    instances: 3
    autoscaling:
      min: 3
      max: 20
      target_cpu: 70%

inference_cluster:
  - type: g5.xlarge
    count: 5 (autoscaling 5-30)
    gpu: NVIDIA A10G (24GB)
    vcpu: 4
    ram: 16GB
    purpose: Real-time inference

  configuration:
    model_serving: ONNX Runtime + TensorRT
    batch_size: 8
    max_latency_ms: 500
    throughput_target: 1000 req/min

training_cluster:
  - type: p4d.24xlarge
    count: 1 (on-demand when training)
    gpu: 8x NVIDIA A100 (40GB)
    vcpu: 96
    ram: 1152GB
    purpose: Weekly model retraining

storage:
  - type: S3
    bucket: fintechco-prod-id-validation
    size: 10TB
    lifecycle:
      - images: 90 days then delete
      - models: retain indefinitely
      - logs: 365 days then archive to Glacier

database:
  - type: Snowflake
    warehouse: ML_INFERENCE_WH
    size: LARGE
    auto_suspend: 5 minutes
    auto_resume: true
```

### Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Route 53 (DNS)                           │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────────────┐
│           CloudFront CDN (Static Assets)                    │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────────────┐
│         Application Load Balancer (ALB)                     │
│  ├─ SSL Termination (TLS 1.3)                               │
│  ├─ WAF (DDoS protection)                                   │
│  └─ Rate Limiting (1000 req/min per IP)                     │
└──────────────────┬──────────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        ↓                     ↓
┌───────────────┐     ┌───────────────┐
│  API Gateway  │     │  API Gateway  │
│   (Primary)   │     │  (Standby)    │
│  us-west-2    │     │  us-east-1    │
└───────┬───────┘     └───────────────┘
        │
        ↓
┌─────────────────────────────────────────────────────────────┐
│          ECS Fargate Cluster (Inference Services)           │
│  ┌─────────────────────────────────────────────────────────┤
│  │ Service: Preprocessing (CPU)                             │
│  │   Tasks: 5-20 (autoscaling)                              │
│  ├─────────────────────────────────────────────────────────┤
│  │ Service: Stage 1 Inference (CPU)                         │
│  │   Tasks: 5-20 (autoscaling)                              │
│  ├─────────────────────────────────────────────────────────┤
│  │ Service: Stage 2 Inference (GPU g5.xlarge)               │
│  │   Tasks: 5-30 (autoscaling)                              │
│  ├─────────────────────────────────────────────────────────┤
│  │ Service: Risk Fusion & Decision (CPU)                    │
│  │   Tasks: 5-20 (autoscaling)                              │
│  └─────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────┘
        │
        ↓
┌─────────────────────────────────────────────────────────────┐
│                   Data Layer                                │
│  ├─ S3: Image storage (encrypted at rest)                   │
│  ├─ Snowflake: Predictions, features, labels                │
│  ├─ Redis: Caching (feature vectors, model outputs)         │
│  └─ MLflow: Model registry                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Security & Compliance

### Data Protection

#### Encryption
- **At Rest:** AES-256 encryption for S3 and Snowflake
- **In Transit:** TLS 1.3 for all API communication
- **PII Data:** ID images encrypted with customer-specific keys (AWS KMS)

#### Access Control
```yaml
roles:
  - name: fraud_analyst
    permissions:
      - read: submissions, predictions, features
      - write: manual_reviews
      - no_access: model_weights, infrastructure

  - name: data_scientist
    permissions:
      - read: submissions, predictions, features, model_weights
      - write: model_registry, training_jobs
      - no_access: customer_pii

  - name: admin
    permissions:
      - read: all
      - write: all
      - delete: model_versions, archived_data

authentication:
  method: JWT
  expiration: 1 hour
  refresh_token: 30 days
  mfa_required: true (for admin, data_scientist)
```

#### Data Retention
```yaml
retention_policy:
  id_images:
    duration: 90 days
    action: permanent_delete
    exception: manual_review_cases (retain 1 year)

  predictions:
    duration: 7 years
    action: archive_to_glacier
    reason: regulatory_compliance

  model_versions:
    duration: indefinite
    action: retain_all
```

### Compliance

#### KYC/AML Requirements
- Document validation meets FFIEC Customer Identification Program guidance
- Audit trail maintained for all validation decisions (7 years)
- Explainability reports generated for regulatory review

#### GDPR/CCPA
- Right to deletion: ID images deleted within 30 days of request
- Data export: Customers can request validation history
- Consent tracking: Explicit consent recorded for image processing

#### Model Governance
- Model versioning with full lineage tracking
- A/B testing for new model rollouts
- Rollback capability within 5 minutes
- Quarterly bias audits (demographic parity, equal opportunity)

---

## Testing Strategy

### Unit Tests
```python
# Example: Test FFT spectral anomaly feature
def test_fft_spectral_anomaly_real_photo():
    """Test that real photos score low on FFT anomaly."""
    real_photo = load_test_image("tests/data/real_drivers_license.jpg")
    score = extract_fft_spectral_anomaly(real_photo)
    assert score < 0.3, f"Real photo scored too high: {score}"

def test_fft_spectral_anomaly_ai_generated():
    """Test that AI images score high on FFT anomaly."""
    ai_photo = load_test_image("tests/data/ai_generated_license.jpg")
    score = extract_fft_spectral_anomaly(ai_photo)
    assert score > 0.6, f"AI photo scored too low: {score}"
```

### Integration Tests
```python
def test_full_pipeline_real_id():
    """Test complete pipeline with authentic ID."""
    response = client.post("/api/v1/id-validation/submit", json={
        "customer_id": "test_customer",
        "id_type": "drivers_license",
        "image": encode_test_image("tests/data/real_ca_license.jpg")
    })

    assert response.status_code == 200
    assert response.json()["decision"] == "approved"
    assert response.json()["risk_score"] < 0.3
    assert response.json()["processing_latency_ms"] < 500

def test_full_pipeline_ai_id():
    """Test complete pipeline with AI-generated ID."""
    response = client.post("/api/v1/id-validation/submit", json={
        "customer_id": "test_customer",
        "id_type": "drivers_license",
        "image": encode_test_image("tests/data/ai_generated_license.jpg")
    })

    assert response.status_code == 200
    assert response.json()["decision"] in ["manual_review", "rejected"]
    assert response.json()["risk_score"] > 0.6
```

### Performance Tests
```bash
# Load test: 1000 req/min sustained for 10 minutes
locust -f tests/load_test.py \
    --host=https://api-staging.fintechco.com \
    --users=1000 \
    --spawn-rate=100 \
    --run-time=10m

# Expected results:
# - Median latency: <200ms
# - p99 latency: <500ms
# - Error rate: <0.1%
```

### Model Validation
```python
def test_model_performance_on_test_set():
    """Validate model meets SLA on held-out test set."""
    test_loader = load_test_dataset("tests/data/test_set_10k.csv")

    predictions = model.predict(test_loader)
    metrics = compute_metrics(predictions, test_loader.labels)

    assert metrics["accuracy"] >= 0.85, "Test accuracy below target"
    assert metrics["recall"] >= 0.85, "Test recall below target"
    assert metrics["fpr"] <= 0.03, "False positive rate too high"
```

---

## Deployment Plan

### Phase 1: Shadow Mode (Weeks 1-4)

**Objective:** Validate detection accuracy in production without blocking traffic

```yaml
configuration:
  mode: shadow
  stage2_enabled: true
  blocking_enabled: false
  logging: verbose
  sample_rate: 100%  # Log all predictions

actions:
  - Deploy Stage 2 models to production
  - Run inference on all submissions (parallel to existing system)
  - Log predictions but do not affect user experience
  - Manual review team labels 500 cases/week

success_criteria:
  - Shadow mode accuracy ≥80% (validated via manual labels)
  - p99 latency <750ms
  - Zero production incidents
  - 2000+ labeled samples collected
```

### Phase 2: Beta (Weeks 5-12)

**Objective:** Enable blocking for high-confidence AI-generated IDs

```yaml
configuration:
  mode: beta
  blocking_threshold: 0.9  # Only block critical risk
  manual_review_threshold: 0.6
  rollout:
    - week_5_6: 10% of traffic
    - week_7_8: 25% of traffic
    - week_9_10: 50% of traffic
    - week_11_12: 100% of traffic

actions:
  - Gradually increase traffic percentage
  - Monitor false positive rate daily
  - Implement A/B testing framework
  - Weekly threshold tuning based on feedback

success_criteria:
  - Synthetic ID fraud reduced by ≥30%
  - False positive rate <3%
  - p99 latency <500ms
  - User abandonment rate no increase
```

### Phase 3: General Availability (Weeks 13+)

**Objective:** Full production rollout with continuous optimization

```yaml
configuration:
  mode: production
  blocking_threshold: 0.8  # Lower threshold based on Beta performance
  manual_review_threshold: 0.5
  traffic: 100%

actions:
  - Enable auto-scaling (5-30 instances)
  - Activate weekly retraining pipeline
  - Implement champion/challenger model testing
  - Enable real-time monitoring dashboards

success_criteria:
  - 50% reduction in synthetic ID fraud losses
  - TPR ≥85%, FPR <3%
  - 99.9% uptime
  - Active learning pipeline processing ≥500 labels/week
```

### Rollback Plan

```yaml
triggers:
  - false_positive_rate > 5%
  - p99_latency > 1000ms
  - error_rate > 1%
  - synthetic_fraud_increase > 20%

rollback_procedure:
  1. Alert on-call engineer (PagerDuty)
  2. Disable Stage 2 (revert to Stage 1 only)
  3. Execute rollback script: ./scripts/rollback_to_previous_version.sh
  4. Verify Stage 1 operating normally
  5. Post-mortem within 24 hours
  6. Fix issue, test, re-deploy

estimated_rollback_time: <5 minutes
```

---

## Monitoring & Observability

### Key Metrics

#### Business Metrics
```yaml
metrics:
  - name: synthetic_id_fraud_losses
    type: gauge
    frequency: daily
    target: <$200K per 6 months
    alert: >$50K/month

  - name: synthetic_id_detection_rate
    type: gauge
    frequency: weekly
    target: ≥85%
    alert: <80%

  - name: false_positive_rate
    type: gauge
    frequency: hourly
    target: <3%
    alert: >5%
```

#### System Metrics
```yaml
metrics:
  - name: api_latency_p50
    type: histogram
    frequency: real-time
    target: <200ms
    alert: >300ms

  - name: api_latency_p99
    type: histogram
    frequency: real-time
    target: <500ms
    alert: >750ms

  - name: throughput
    type: counter
    frequency: real-time
    target: 1000 req/min sustained
    alert: <500 req/min during peak

  - name: error_rate
    type: gauge
    frequency: real-time
    target: <0.1%
    alert: >1%

  - name: gpu_utilization
    type: gauge
    frequency: 1 minute
    target: 70-85%
    alert: >95% or <30%
```

#### Model Metrics
```yaml
metrics:
  - name: model_accuracy
    type: gauge
    frequency: weekly
    target: ≥85%
    alert: <80%
    computation: validated_against_manual_labels

  - name: model_drift
    type: gauge
    frequency: weekly
    target: <0.1 PSI (Population Stability Index)
    alert: >0.2 PSI

  - name: feature_importance_shift
    type: gauge
    frequency: monthly
    target: <20% change in top-5 features
    alert: >30% change
```

### Dashboards

#### Operations Dashboard (Grafana)
```
┌─────────────────────────────────────────────────────────────┐
│  ID Validation System - Operations Dashboard                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  [Real-time Traffic]         [Latency Distribution]         │
│   Current: 347 req/min        p50: 142ms  p99: 387ms       │
│   Peak: 892 req/min                                         │
│                                                              │
│  [Decision Breakdown - Last 24h]                            │
│   Auto-Approved: 87.2%                                      │
│   Manual Review: 8.3%                                       │
│   Auto-Rejected: 4.5%                                       │
│                                                              │
│  [Error Rate]                [GPU Utilization]              │
│   Current: 0.03%              GPU-1: 73%   GPU-2: 68%      │
│   Errors/hour: 2              GPU-3: 71%   GPU-4: 69%      │
│                                                              │
│  [Recent Alerts]                                            │
│   ⚠ p99 latency spike at 14:32 UTC (resolved)              │
│   ✓ All systems operational                                 │
└─────────────────────────────────────────────────────────────┘
```

#### Fraud Analytics Dashboard
```
┌─────────────────────────────────────────────────────────────┐
│  Synthetic ID Fraud Analytics                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  [Fraud Loss Trend - Last 6 Months]                         │
│   Pre-deployment: $420K total                               │
│   Post-deployment: $187K total (55% reduction)              │
│                                                              │
│  [Detection Performance]                                     │
│   TPR: 87.3%   FPR: 2.1%   Accuracy: 91.2%                  │
│                                                              │
│  [AI Detection Breakdown]                                    │
│   Traditional Forgery: 12%                                  │
│   AI-Generated: 73%                                         │
│   Deepfake Face: 15%                                        │
│                                                              │
│  [Manual Review Queue]                                      │
│   Pending: 47 cases                                         │
│   Avg Review Time: 2.3 minutes                              │
│   Backlog: 0.8 hours                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Appendices

### Appendix A: Hardware Requirements Summary

| Component | Development | Production |
|-----------|-------------|------------|
| GPU | 1x A10G (24GB) | 5-30x A10G (autoscaling) |
| CPU | 4 vCPU | 4 vCPU per instance |
| RAM | 16GB | 16GB per instance |
| Storage | 100GB S3 | 10TB S3 + Snowflake |
| Estimated Cost | $500/month | $8,000-$20,000/month |

### Appendix B: Training Data Sources

| Source | Description | Size | Cost |
|--------|-------------|------|------|
| DMV Partnership | Authentic US driver's licenses | 50K images | Legal agreement |
| Passport Database | Government partnership | 30K images | Government contract |
| RAISE Dataset | Authentic photos (camera fingerprints) | 8K images | Free (academic) |
| FaceForensics++ | Deepfake detection benchmark | 1M frames | Free (academic) |
| Internal Synthetic | AI-generated IDs (SD, MJ, DALL-E) | 50K images | GPU compute cost |

### Appendix C: Model Versioning Strategy

```
Model Naming Convention: {model_name}_v{major}.{minor}.{patch}

Examples:
  - ai_artifact_detector_v1.0.0 (Initial release)
  - ai_artifact_detector_v1.1.0 (Feature update)
  - ai_artifact_detector_v1.1.1 (Bug fix)
  - ai_artifact_detector_v2.0.0 (Architecture change)

Version Control:
  - All models stored in MLflow Registry
  - Git tags for reproducibility
  - Docker images for deployment
  - A/B testing for new versions (champion vs. challenger)
```

### Appendix D: Related Documents

- [Product Requirements Document](./prd-ai-generated-document-detection.md)
- [Macro Risk Assessment Dec 2025](../reports/documents/macro-risk-assessment-dec2025-validation.md)
- [Existing ID Validation Model Code](../scripts/id_card_validation_model.py)

---

## Approval & Review

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Engineering Lead | [TBD] | | |
| Data Science Lead | [TBD] | | |
| Infrastructure Lead | [TBD] | | |
| InfoSec Lead | [TBD] | | |

---

**Document History:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-14 | Claude Code | Initial technical specification |

---

**Next Steps:**
1. Engineering review and approval
2. Infrastructure provisioning (AWS resources)
3. Training data acquisition (DMV partnership)
4. Begin Phase 1: Proof of Concept development
