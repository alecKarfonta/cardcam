# CVAT Trading Card Annotation Guidelines

## Overview

This document provides comprehensive guidelines for annotating trading card images using CVAT (Computer Vision Annotation Tool). The goal is to create high-quality training data for instance segmentation models that can accurately detect and segment individual trading cards in various scenarios.

## Annotation Objectives

- **Primary Goal**: Create pixel-perfect segmentation masks for individual trading cards
- **Target Accuracy**: >95% IoU consistency between annotators
- **Class Definition**: Single "card" class with instance-level separation
- **Output Format**: COCO format for instance segmentation

## General Annotation Principles

### 1. Precision Requirements
- **Boundary Accuracy**: Follow card edges precisely, including rounded corners
- **Pixel-Perfect**: Zoom in to ensure accurate boundary placement
- **Consistency**: Maintain consistent annotation quality across all images

### 2. Class Definition
- **Single Class**: All trading cards are labeled as "card" regardless of game type
- **Instance Separation**: Each individual card gets its own instance annotation
- **No Sub-classes**: Do not differentiate between Magic, Pokémon, Yu-Gi-Oh!, etc.

## Specific Annotation Scenarios

### Single Card Images
- **Complete Visibility**: Annotate the entire visible card boundary
- **Partial Cards**: If >70% of card is visible, annotate the visible portion
- **Skip Criteria**: Skip cards with <70% visibility

### Multi-Card Layouts
- **Individual Instances**: Each card gets a separate annotation instance
- **Overlapping Cards**: Annotate visible portions only, do not extrapolate hidden areas
- **Grid Arrangements**: Annotate each card in the grid individually

### Complex Backgrounds
- **Binder Pages**: Include card boundaries, exclude binder holes/rings
- **Protective Sleeves**: Annotate card boundary, not sleeve boundary
- **Table Surfaces**: Focus on card edges, ignore shadows/reflections

## Edge Cases and Special Situations

### Damaged Cards
- **Torn Cards**: Annotate actual card boundary, including damaged edges
- **Bent Cards**: Follow the visible card outline, including bent portions
- **Water Damage**: Annotate if card structure is still recognizable

### Lighting and Reflections
- **Glossy Surfaces**: Annotate card boundary, ignore surface reflections
- **Flash Photography**: Focus on actual card edges, not light artifacts
- **Shadows**: Include card area, exclude cast shadows

### Partial Occlusion
- **Hand Coverage**: Annotate visible card portions only
- **Object Overlap**: Do not extrapolate hidden card areas
- **Other Cards**: Annotate each visible card separately

## Quality Control Standards

### Annotation Accuracy
- **Boundary Precision**: ±2 pixels tolerance for card edges
- **Corner Handling**: Properly capture rounded corners on modern cards
- **Straight Edges**: Ensure straight lines for card sides

### Consistency Checks
- **Inter-annotator Agreement**: Target >95% IoU between different annotators
- **Self-consistency**: Review your own annotations for consistency
- **Edge Case Documentation**: Note unusual cases for team discussion

## CVAT-Specific Instructions

### Project Setup
1. **Create Project**: Use "Instance Segmentation" task type
2. **Label Configuration**: Single label "card" with polygon annotation
3. **Image Upload**: Batch upload 50-100 images per task

### Annotation Workflow
1. **Initial Review**: Examine image for all visible cards
2. **Polygon Tool**: Use polygon tool for precise boundary annotation
3. **Zoom and Refine**: Zoom in to refine boundary accuracy
4. **Instance Separation**: Ensure each card has separate instance ID

### Keyboard Shortcuts
- **N**: Next image
- **P**: Previous image
- **Ctrl+Z**: Undo last action
- **Space**: Pan mode toggle
- **+/-**: Zoom in/out

## Annotation Examples

### Good Annotations ✅
- Precise boundary following card edges
- Proper handling of rounded corners
- Separate instances for each card
- Consistent annotation across similar scenarios

### Poor Annotations ❌
- Rough/approximate boundaries
- Missing corner details
- Merged instances for separate cards
- Inconsistent handling of similar cases

## Quality Assurance Process

### Self-Review Checklist
- [ ] All visible cards annotated
- [ ] Boundaries follow card edges precisely
- [ ] Each card has separate instance
- [ ] Rounded corners properly captured
- [ ] No extrapolation of hidden areas

### Peer Review Process
1. **Random Sampling**: 10% of annotations reviewed by peers
2. **IoU Calculation**: Measure overlap accuracy
3. **Feedback Loop**: Discuss and resolve discrepancies
4. **Continuous Improvement**: Update guidelines based on findings

## Performance Metrics

### Target Metrics
- **Annotation Speed**: 2-3 minutes per single card image
- **Multi-card Speed**: 5-10 minutes per image (depending on card count)
- **Accuracy**: >95% IoU on validation samples
- **Consistency**: <5% variation between annotators

### Progress Tracking
- **Daily Targets**: 50-100 images per annotator per day
- **Quality Metrics**: Track IoU scores and consistency
- **Completion Rate**: Monitor annotation completion percentage

## Troubleshooting

### Common Issues
- **Blurry Boundaries**: Use maximum zoom for precision
- **Overlapping Cards**: Focus on visible portions only
- **Reflective Surfaces**: Ignore reflections, focus on card structure
- **Unusual Angles**: Follow visible card outline regardless of perspective

### Technical Support
- **CVAT Issues**: Check Docker container status
- **Performance**: Reduce image batch size if CVAT is slow
- **Data Export**: Use COCO format for training data export

## Data Export and Validation

### Export Format
- **Primary**: COCO format for instance segmentation
- **Backup**: YOLO format for object detection fallback
- **Validation**: Export 15% of data for validation set

### Post-Export Validation
1. **Format Check**: Verify COCO JSON structure
2. **Annotation Count**: Confirm all instances exported
3. **Boundary Quality**: Spot-check annotation accuracy
4. **Dataset Split**: Ensure proper train/val/test distribution

## Continuous Improvement

### Feedback Integration
- **User Reports**: Collect feedback on annotation quality
- **Model Performance**: Use model results to identify annotation issues
- **Guideline Updates**: Regularly update guidelines based on learnings

### Training and Onboarding
- **New Annotator Training**: 2-hour training session with examples
- **Practice Set**: 50 practice annotations before production work
- **Ongoing Support**: Regular check-ins and guidance sessions

---

**Version**: 1.0  
**Last Updated**: September 26, 2025  
**Contact**: Development Team for questions and clarifications
