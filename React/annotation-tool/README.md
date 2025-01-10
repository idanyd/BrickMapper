# Image Annotation Tool

A React-based web application for annotating images with bounding boxes. This tool allows users to load multiple images and create labeled annotations, perfect for creating training data for machine learning models or marking specific regions in images.

## Features

- Upload and navigate through multiple images
- Draw bounding boxes with two different label types:
  - Step Box (Green, Label 1)
  - Step Number (Blue, Label 2)
- Keyboard shortcuts for efficient annotation
- Export annotations in JSON format
- Undo last annotation
- Clear all annotations for current image

## Usage

### Getting Started

1. Click the folder upload button to select a directory containing images
2. Use the interface to navigate through images and create annotations

### Controls

- **Navigation**:
  - Left Arrow or 'A': Previous image
  - Right Arrow or 'D': Next image
  - '1': Select Step Box annotation type
  - '2': Select Step Number annotation type

- **Drawing**:
  - Click and drag on the image to draw a bounding box
  - Release to complete the annotation

### Annotation Export

Annotations are exported as a JSON file containing:
- Image path
- Bounding box coordinates
- Label types

## Technical Requirements

- React 16.8+ (uses hooks)
- Modern web browser with HTML5 Canvas support

## Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev