
# BrickMapper

BrickMapper is a comprehensive machine learning project that helps LEGO^®^ enthusiasts identify and locate where  LEGO^®^ pieces are used within instruction manuals. The project combines computer vision, deep learning, and web technologies to create an intelligent system for LEGO^®^ piece recognition and step tracking.

## 🎯 Project Overview

**Goal**: Develop a machine learning model to identify LEGO pieces in instruction manuals and track which steps they appear in.

**Current Status**: Production-ready with proof of concept completed on 40+ LEGO sets.

**Live Application**: [brockmapper.netlify.com](https://brockmapper.netlify.com)

## 🚀 Features

### Web Application
- **Element Search**: Search for LEGO elements by their unique element ID
- **Step Locator**: Find exactly which steps in which instruction booklets contain a specific element
- **PDF Links**: Direct links to the specific pages in official LEGO instruction PDFs
- **Element Details**: View information about elements including part number, name, color, and image

### Machine Learning Pipeline
- **YOLO11n Model**: Automated detection of step boxes in instruction manuals
- **Computer Vision**: Template matching and piece identification using OpenCV
- **GPU Acceleration**: CUDA-optimized processing for significantly faster performance
- **Multi-threaded Processing**: Efficient CPU fallback with configurable worker threads

### Data Processing
- **PDF Processing**: Automated extraction of images and structured data from LEGO instruction PDFs
- **Database Integration**: Comprehensive data management with SQLite/Supabase support

## 🏗️ Architecture

### Technology Stack

**Frontend**:
- React with Tailwind CSS for styling
- Vite for build tooling
- Deployed on Netlify

**Backend & Database**:
- Supabase for production database and authentication
- SQLite for local development
- PostgreSQL-compatible schema

**Machine Learning**:
- YOLO11n for object detection
- OpenCV for computer vision and template matching
- CUDA support for GPU acceleration
- Neptune for experiment tracking

**Data Processing**:
- PyMuPDF for PDF processing
- Pandas for data manipulation
- PIL/Pillow for image processing

## 📊 Database Schema

The application uses the following database structure:

### Core Tables (Created by Rebrickable^®^)
- **elements**: LEGO element information (ID, part number, name, color, image URL)
- **sets**: LEGO set information (set number, name, year, image URL)
- **inventories**: Set inventory data linking sets to their elements
- **inventory_parts**: Individual parts within set inventories

### Step Tracking Tables
- **set_steps**: Maps steps to specific pages in instruction booklets
- **step_elements**: Links elements to the steps where they appear

## 🛠️ Setup and Installation

### Prerequisites
- Python 3.8+
- Node.js 16+
- CUDA-compatible GPU (recommended for optimal performance)

### Backend Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/idanyd/BrickMapper.git
   cd BrickMapper
   ```

2. Create a virtual environment:
   ```bash
   python -m venv brickmapper
   source brickmapper/bin/activate  # On Windows: brickmapper\Scripts\activate
   ```

3. Install Python dependencies:
   ```bash
   pip install -r reqs.txt
   ```

4. Set up environment variables:
   ```bash
   # Create .env file
   DATABASE_URL=your_database_url
   NEPTUNE_PROJECT=your_neptune_project
   NEPTUNE_API_TOKEN=your_neptune_token
   ```

### Frontend Setup

1. Navigate to the React application:
   ```bash
   cd React/site
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Create environment file:
   ```bash
   # Create .env file
   VITE_SUPABASE_URL=your_supabase_url
   VITE_SUPABASE_KEY=your_supabase_key
   ```

4. Start development server:
   ```bash
   npm run dev
   ```

## 🔬 Machine Learning Pipeline

### Step Detection
The project uses a YOLO11n model trained specifically for detecting step boxes in LEGO instruction manuals:

```python
from src.step_detector import StepDetector

# Initialize detector
detector = StepDetector(manual_doc)

# Detect steps in an image
steps = detector.detect_steps(image, page_num, conf_threshold=0.25)
```

### Piece Matching
Advanced template matching with GPU acceleration:

```python
from src.piece_matcher import PieceMatcher

# Initialize matcher
matcher = PieceMatcher()

# Load pieces and perform matching
matcher.load_step_pieces_from_directory(step_pieces_dir)
matcher.load_set_pieces_from_directory(set_pieces_dir)
matched, unmatched = matcher.match_pieces(threshold=0.031)
```

### Performance Optimization
- **GPU Processing**: CUDA streams for parallel template matching
- **Multi-threading**: Configurable worker threads for CPU processing
- **Memory Management**: Efficient GPU memory handling with automatic cleanup

## 📁 Project Structure

```
BrickMapper/
├── src/                          # Core Python modules
│   ├── step_detector.py         # YOLO-based step detection
│   ├── piece_matcher.py         # Template matching and piece identification
│   ├── pdf_processor.py         # PDF processing utilities
│   ├── data_loader.py           # Database operations
│   ├── dataset.py               # Dataset management
│   └── utils/                   # Utility functions
├── React/site/                  # Frontend application
│   ├── src/                     # React components
│   └── public/                  # Static assets
├── notebooks/                   # Jupyter notebooks for analysis
│   ├── 01_data_loading.ipynb   # Data loading examples
│   ├── 02_pdf_processing.ipynb # PDF processing workflow
│   └── 03_element_step_finder.ipynb # Element analysis
├── tests/                       # Unit tests
└── data/                        # Data storage (not in repo)
```

## 🚀 Usage Examples

### Processing a New LEGO Set

```python
from src.pdf_processor import PDFProcessor
from src.step_detector import StepDetector

# Process PDF manual
processor = PDFProcessor("path/to/manual.pdf")
images = processor.extract_images()

# Detect steps
detector = StepDetector(processor.doc)
for page_num, image in enumerate(images):
    steps = detector.detect_steps(image, page_num)
    print(f"Found {len(steps)} steps on page {page_num}")
```

### Finding Elements in Steps

```python
from src.db_interface import DatabaseInterface

# Initialize database connection
db = DatabaseInterface(database_url)

# Find all steps containing a specific element
element_steps = db.find_element_steps("element_id_here")
for step in element_steps:
    print(f"Element found in Set {step.set_num}, Page {step.page}, Step {step.step}")
```

## 📈 Current Status

### Completed Features
- ✅ YOLO11n model training and deployment
- ✅ GPU-accelerated template matching
- ✅ PDF processing pipeline
- ✅ React frontend with Supabase integration
- ✅ Database schema and data loading
- ✅ Proof of concept with 40+ processed sets

### Performance Metrics
- **Processing Speed**: GPU processing is significantly faster than CPU
- **Model Accuracy**: YOLO11n provides reliable step box detection
- **Template Matching**: Effective piece identification with configurable thresholds
- **Data Coverage**: 40+ LEGO sets processed as proof of concept

### Recommended Setup
- **GPU**: CUDA-compatible GPU for optimal performance
- **Memory**: 8GB+ RAM recommended for large manual processing
- **Storage**: SSD recommended for faster image processing

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test modules
python -m pytest tests/test_dataset.py
python -m pytest tests/test_data_loader.py
```

## 📊 Monitoring and Logging

The project includes comprehensive logging and experiment tracking:

- **Neptune Integration**: Track model training and evaluation metrics
- **Structured Logging**: Detailed logging throughout the pipeline
- **Performance Profiling**: Built-in profiling for optimization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the terms of the  [GPL V3.0 license](https://github.com/idanyd/BrickMapper?tab=GPL-3.0-1-ov-file).

## 🙏 Acknowledgments

- LEGO^®^ Group for providing instruction manuals and piece data
- Rebrickable^®^ for creating and maintaining the [LEGO^®^ catalog database](https://rebrickable.com/downloads/)
- [Brickognize](https://brickognize.com/) for providing the LEGO pieces image recognition API
- Ultralytics for the YOLO framework
- OpenCV community for computer vision tools
- Supabase for backend infrastructure
- Netlify for frontend infrastructure

## 📞 Support

For questions, issues, or contributions, please:
- Open an issue on GitHub

---

**Note**: This project is for educational and research purposes. LEGO^®^ is a trademark of the LEGO Group.
