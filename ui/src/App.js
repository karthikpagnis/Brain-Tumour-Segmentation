import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// ImageUploader Component
const ImageUploader = ({ onUpload }) => {
  const [dragActive, setDragActive] = useState(false);
  const [file, setFile] = useState(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.name.endsWith('.nii') || droppedFile.name.endsWith('.nii.gz')) {
        setFile(droppedFile);
        onUpload(droppedFile);
      } else {
        alert('Please upload a NIfTI file (.nii or .nii.gz)');
      }
    }
  };

  const handleChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      onUpload(selectedFile);
    }
  };

  return (
    <div className="uploader">
      <div
        className={`drop-zone ${dragActive ? 'active' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <svg className="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
          <polyline points="17 8 12 3 7 8"></polyline>
          <line x1="12" y1="3" x2="12" y2="15"></line>
        </svg>
        <p className="upload-text">
          Drag and drop your MRI scan (.nii.gz)
        </p>
        <p className="upload-subtext">or</p>
        <label className="file-label">
          <span>Browse Files</span>
          <input
            type="file"
            accept=".nii,.nii.gz"
            onChange={handleChange}
            style={{ display: 'none' }}
          />
        </label>
      </div>
      {file && <p className="file-selected">✓ File selected: {file.name}</p>}
    </div>
  );
};

// MRIViewer Component
const MRIViewer = ({ imageData }) => {
  const [sliceIndex, setSliceIndex] = useState(0);

  if (!imageData) {
    return <div className="viewer-placeholder">Upload an MRI scan to view</div>;
  }

  const depth = imageData.shape[0];
  const height = imageData.shape[1];
  const width = imageData.shape[2];

  return (
    <div className="viewer">
      <div className="viewer-info">
        <p>Depth: {sliceIndex + 1} / {depth}</p>
        <p>Dimensions: {height}x{width}</p>
      </div>
      <div className="viewer-controls">
        <input
          type="range"
          min="0"
          max={depth - 1}
          value={sliceIndex}
          onChange={(e) => setSliceIndex(parseInt(e.target.value))}
          className="slider"
        />
      </div>
    </div>
  );
};

// ResultsDisplay Component
const ResultsDisplay = ({ predictions, downloadUrl }) => {
  if (!predictions) return null;

  return (
    <div className="results">
      <div className="results-header">
        <h3>Segmentation Results</h3>
        <p>Class distribution:</p>
      </div>
      <div className="class-info">
        <div className="class-item">
          <span className="class-color" style={{ backgroundColor: '#000' }}></span>
          <span>Background</span>
        </div>
        <div className="class-item">
          <span className="class-color" style={{ backgroundColor: '#FF0000' }}></span>
          <span>Necrotic Core</span>
        </div>
        <div className="class-item">
          <span className="class-color" style={{ backgroundColor: '#00FF00' }}></span>
          <span>Peritumoral Edema</span>
        </div>
        <div className="class-item">
          <span className="class-color" style={{ backgroundColor: '#0000FF' }}></span>
          <span>Enhancing Tumor</span>
        </div>
      </div>
      {downloadUrl && (
        <a href={downloadUrl} download className="download-btn">
          ⬇ Download NIfTI Result
        </a>
      )}
    </div>
  );
};

// Main App Component
export default function App() {
  const [file, setFile] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [processingTime, setProcessingTime] = useState(null);
  const [downloadUrl, setDownloadUrl] = useState(null);

  useEffect(() => {
    // Check API health on mount
    axios
      .get(`${API_URL}/api/health`)
      .then(() => console.log('✓ API connected'))
      .catch(() => setError('Could not connect to API server'));
  }, []);

  const handleUpload = async (uploadedFile) => {
    setFile(uploadedFile);
    setPredictions(null);
    setError(null);
    setLoading(true);

    try {
      const formData = new FormData();
      formData.append('file', uploadedFile);

      const response = await axios.post(`${API_URL}/api/predict`, formData);

      if (response.data.status === 'success') {
        setPredictions(response.data);
        setProcessingTime(response.data.processing_time);
        setDownloadUrl(`${API_URL}${response.data.predictions_url}`);
      } else {
        setError('Prediction failed');
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Error during prediction');
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>🧠 Brain Tumor Segmentation</h1>
        <p>AI-powered MRI analysis using Attention-Enhanced U-Net</p>
      </header>

      <main className="app-main">
        <section className="upload-section">
          <h2>Step 1: Upload MRI Scan</h2>
          <ImageUploader onUpload={handleUpload} />
        </section>

        {loading && (
          <section className="loading-section">
            <div className="spinner"></div>
            <p>Analyzing MRI scan... This may take a minute</p>
          </section>
        )}

        {error && (
          <section className="error-section">
            <p>⚠️ Error: {error}</p>
          </section>
        )}

        {processingTime && (
          <section className="info-section">
            <p>✓ Processing completed in {processingTime.toFixed(2)} seconds</p>
          </section>
        )}

        <section className="visualization-section">
          <h2>Step 2: View Results</h2>
          <MRIViewer imageData={predictions} />
          <ResultsDisplay predictions={predictions} downloadUrl={downloadUrl} />
        </section>

        <section className="info-section">
          <h3>About This Model</h3>
          <ul>
            <li>✨ Attention-Enhanced U-Net architecture</li>
            <li>🎯 4-class tumor segmentation</li>
            <li>📊 DSC, IoU metrics for validation</li>
            <li>🔬 Research-grade implementation</li>
          </ul>
        </section>
      </main>

      <footer className="app-footer">
        <p>Brain Tumor Segmentation | IIT Madras 2025</p>
      </footer>
    </div>
  );
}
