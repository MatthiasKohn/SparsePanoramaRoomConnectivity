# Automatic Semantic Room Connectivity from Sparse 360° Panoramas

## Project Summary

**PhD Topic:** 3D Scene Perception and Change Detection

### Goal
Recover room connectivity and relative room poses from sparse 360° panoramas without requiring dense overlap or traditional SfM feature matching.

### Motivation
In many indoor environments, panoramas are captured sparsely, often with little or no direct overlap. The main shared geometric evidence between rooms exists through doorways and openings.

### Input
- Sparse 360° panoramas
- Monocular depth estimates (currently Depth Anywhere Panoramas)
- Optional doorway detections

### Desired Output
- Room connectivity graph
- Relative room poses
- Coarse 3D structure
- Foundation for future change detection

---

## Current Core Hypothesis

Doorways act as geometric apertures.

Pixels inside a doorway correspond to rays that enter neighboring rooms. The surfaces observed through the doorway are potentially visible from both rooms and therefore provide geometric constraints for pose estimation.

---

## Current Pipeline


---

## Current Research Stages

### Stage 1
Known-pose sanity check

### Stage 2
Pose recovery from perturbed initialization

### Stage 3
Unknown pose estimation

### Stage 4
Automatic doorway detection and scaling

---

## Success Criteria

- Robustness to sparse overlap
- Works on realistic indoor environments
- Scales beyond hand-selected examples
