import { useState, useEffect, useRef } from 'react'

function App() {
  const [images, setImages] = useState([]); // Array of all images
  const [currentImageIndex, setCurrentImageIndex] = useState(0); // Current image being displayed
  const [fileNames, setFileNames] = useState([]); // Array of file names

  const [image, setImage] = useState(null);
  const [annotations, setAnnotations] = useState([]);
  const [drawing, setDrawing] = useState(false);
  const [currentBox, setCurrentBox] = useState(null);
  const [currentLabel, setCurrentLabel] = useState(1);
  const canvasRef = useRef(null);

  const handleFolderUpload = (event) => {
    // Clear existing annotations when loading new folder
    setAnnotations([]);
    setCurrentImageIndex(0);

    const files = Array.from(event.target.files).filter(file => 
      file.type.startsWith('image/')
    );
    
    setFileNames(files.map(file => file.name));
    
    // Load all images
    files.forEach((file, index) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
          setImages(prev => {
            const newImages = [...prev];
            newImages[index] = img;
            return newImages;
          });
        };
        img.src = e.target.result;
      };
      reader.readAsDataURL(file);
    });
  };

  const navigateImages = (direction) => {
    const newIndex = currentImageIndex + direction;
    let targetIndex;

    if (newIndex >= images.length) {
      // Wrap to first image
      targetIndex = 0;
    } else if (newIndex < 0) {
      // Wrap to last image
      targetIndex = images.length - 1;
    } else {
      targetIndex = newIndex;
    }

    setCurrentImageIndex(newIndex);
    
    // Wait for state update to complete using setTimeout
    setTimeout(() => {
      const canvas = canvasRef.current;
      const currentImage = images[newIndex];
      
      // Update canvas dimensions and draw new image
      if (currentImage) {
        canvas.width = currentImage.width;
        canvas.height = currentImage.height;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(currentImage, 0, 0);
        
        // Draw annotations for current image
        annotations
          .filter(ann => ann.imageIndex === newIndex)
          .forEach(({box, label}) => {
            ctx.strokeStyle = label === 1 ? '#00ff00' : '#0000ff';
            ctx.lineWidth = 2;
            ctx.strokeRect(box[0], box[1], box[2] - box[0], box[3] - box[1]);
          });
      }
    }, 0);
  };

  const startDrawing = (e) => {
    e.stopPropagation();
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    setDrawing(true);
    setCurrentBox({
      startX: x,
      startY: y,
      endX: x,
      endY: y
    });
  };

  const draw = (e) => {
    e.stopPropagation();
    if (!drawing) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    setCurrentBox(prev => ({
      ...prev,
      endX: x,
      endY: y
    }));

    drawCanvas();
  };

  const stopDrawing = () => {
    if (!drawing) return;

    setDrawing(false);
    if (currentBox) {
      const newAnnotation = {
        imageIndex: currentImageIndex,
        box: [
          Math.min(currentBox.startX, currentBox.endX),
          Math.min(currentBox.startY, currentBox.endY),
          Math.max(currentBox.startX, currentBox.endX),
          Math.max(currentBox.startY, currentBox.endY)
        ],
        label: currentLabel
      };
      setAnnotations([...annotations, newAnnotation]);
    }
    setCurrentBox(null);
    drawCanvas();
  };

  const drawCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const currentImage = images[currentImageIndex];

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw current image
    if (currentImage) {
      ctx.drawImage(currentImage, 0, 0);
    }

    // Draw annotations for current image only
    annotations
      .filter(ann => ann.imageIndex === currentImageIndex)
      .forEach(({box, label}) => {
        ctx.strokeStyle = label === 1 ? '#00ff00' : '#0000ff';
        ctx.lineWidth = 2;
        ctx.strokeRect(box[0], box[1], box[2] - box[0], box[3] - box[1]);
      });

    // Draw current box
    if (currentBox) {
      ctx.strokeStyle = currentLabel === 1 ? '#00ff00' : '#0000ff';
      ctx.lineWidth = 2;
      ctx.strokeRect(
        currentBox.startX,
        currentBox.startY,
        currentBox.endX - currentBox.startX,
        currentBox.endY - currentBox.startY
      );
    }
  };

  const exportAnnotations = () => {
    const exportData = images.map((img, index) => ({
      image_path: fileNames[index],
      boxes: annotations.filter(a => a.imageIndex === index).map(a => a.box),
      labels: annotations.filter(a => a.imageIndex === index).map(a => a.label)
    }));

    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: 'application/json'
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'annotations.json';
    a.click();
  };

  const undoLastBox = () => {
    // Find the last annotation for the current image
    const currentImageAnnotations = annotations.filter(a => a.imageIndex === currentImageIndex);
    if (currentImageAnnotations.length > 0) {
      // Remove the last annotation for the current image
      const lastAnnotationIndex = annotations.lastIndexOf(currentImageAnnotations[currentImageAnnotations.length - 1]);
      const newAnnotations = [
        ...annotations.slice(0, lastAnnotationIndex),
        ...annotations.slice(lastAnnotationIndex + 1)
      ];
      setAnnotations(newAnnotations);
      drawCanvas();
    }
  };

  const clearAllBoxes = () => {
    setAnnotations(annotations.filter(a => a.imageIndex !== currentImageIndex));
    drawCanvas();
  };

  useEffect(() => {
    drawCanvas();
  }, [annotations, image]);

  useEffect(() => {
    const handleKeyPress = (e) => {
      switch(e.key) {
        case '1':
          setCurrentLabel(1);
          break;
        case '2':
          setCurrentLabel(2);
          break;
        case 'ArrowLeft':
        case 'a':
          navigateImages(-1);
          break;
        case 'ArrowRight':
        case 'd':
          navigateImages(1);
          break;
        
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [currentImageIndex, images, annotations]);

  // Add useEffect to handle image changes
  useEffect(() => {
    if (images[currentImageIndex]) {
      const canvas = canvasRef.current;
      canvas.width = images[currentImageIndex].width;
      canvas.height = images[currentImageIndex].height;
      drawCanvas();
    }
  }, [currentImageIndex, images]);
  return (
    <div className="flex flex-col items-center p-4 max-w-4xl mx-auto">
      <div className="space-y-4 w-full">
        <div className="flex space-x-4 flex-wrap gap-2">
          <input
            type="file"
            webkitdirectory="true"
            directory="true"
            multiple
            onChange={handleFolderUpload}
            className="border p-2"
          />
          
          <select
            value={currentLabel}
            onChange={(e) => setCurrentLabel(parseInt(e.target.value))}
            className="border p-2"
          >
            <option value={1}>Step Box (1)</option>
            <option value={2}>Step Number (2)</option>
          </select>

          <div className="flex space-x-2">
            <button
              onClick={() => navigateImages(-1)}
              className="bg-gray-500 text-white px-4 py-2 rounded"
              disabled={currentImageIndex === 0}
            >
              Previous
            </button>
            <button
              onClick={() => navigateImages(1)}
              className="bg-gray-500 text-white px-4 py-2 rounded"
              disabled={currentImageIndex === images.length - 1}
            >
              Next
            </button>
          </div>

          <button
            onClick={undoLastBox}
            className="bg-yellow-500 text-white px-4 py-2 rounded"
            disabled={!annotations.some(a => a.imageIndex === currentImageIndex)}
          >
            Undo Last Box
          </button>
          
          <button
            onClick={clearAllBoxes}
            className="bg-red-500 text-white px-4 py-2 rounded"
            disabled={!annotations.some(a => a.imageIndex === currentImageIndex)}
          >
            Clear All
          </button>

          <button
            onClick={exportAnnotations}
            className="bg-blue-500 text-white px-4 py-2 rounded"
            disabled={annotations.length === 0}
          >
            Export Annotations
          </button>
        </div>

        {/* Image counter */}
        <div className="text-sm text-gray-600">
          Image {currentImageIndex + 1} of {images.length}
        </div>

        {/* Canvas */}
        <canvas
          ref={canvasRef}
          onMouseDown={startDrawing}
          onMouseMove={draw}
          onMouseUp={stopDrawing}
          onMouseLeave={stopDrawing}
          className="border border-gray-300"
        />

        {/* Instructions */}
        <div className="text-sm text-gray-600">
          <p>Green boxes: Step boxes (Press '1')</p>
          <p>Blue boxes: Step numbers (Press '2')</p>
          <p>Navigate images: Arrow keys or 'A'/'D' keys</p>
          <p>Current number of annotations: {annotations.filter(a => a.imageIndex === currentImageIndex).length}</p>
        </div>
      </div>
    </div>
  );
}

export default App;