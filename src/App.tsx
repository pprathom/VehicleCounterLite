import React, { useState, useRef, useEffect, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import { 
  Camera, 
  Upload, 
  Play, 
  Pause, 
  Download, 
  Trash2, 
  Settings, 
  BarChart3, 
  Car, 
  Truck, 
  Bus, 
  Bike,
  Plus,
  X,
  ArrowDownLeft,
  ArrowUpRight
} from 'lucide-react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

// Utility for tailwind classes
function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// Types
interface Point {
  x: number;
  y: number;
}

interface Line {
  p1: Point;
  p2: Point;
}

interface Detection {
  bbox: [number, number, number, number];
  class: string;
  score: number;
  id?: number;
  centroid: Point;
}

interface TrackedObject {
  id: number;
  class: string;
  lastCentroid: Point;
  path: Point[];
  lastSeen: number;
  counted: boolean;
}

interface CountData {
  car: number;
  truck: number;
  bus: number;
  motorcycle: number;
  other: number;
}

interface HistoryItem {
  timestamp: string;
  type: string;
  id: number;
  direction: 'inbound' | 'outbound';
}

const VEHICLE_CLASSES = ['car', 'truck', 'bus', 'motorcycle'];

export default function App() {
  // Refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const modelRef = useRef<cocoSsd.ObjectDetection | null>(null);
  const requestRef = useRef<number>(null);
  const trackedObjectsRef = useRef<TrackedObject[]>([]);
  const nextIdRef = useRef(1);
  const lastDetectionsRef = useRef<cocoSsd.DetectedObject[]>([]);

  // State
  const [isLoading, setIsLoading] = useState(true);
  const [isPlaying, setIsPlaying] = useState(false);
  const [source, setSource] = useState<'camera' | 'file' | 'stream' | null>(null);
  const [inboundCounts, setInboundCounts] = useState<CountData>({ car: 0, truck: 0, bus: 0, motorcycle: 0, other: 0 });
  const [outboundCounts, setOutboundCounts] = useState<CountData>({ car: 0, truck: 0, bus: 0, motorcycle: 0, other: 0 });
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [line, setLine] = useState<Line | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [drawingStart, setDrawingStart] = useState<Point | null>(null);
  const [previewPoint, setPreviewPoint] = useState<Point | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [showStreamInput, setShowStreamInput] = useState(false);
  const [streamUrl, setStreamUrl] = useState('');
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.4); // Lowered default
  const [debugMode, setDebugMode] = useState(false);
  const [modelBase, setModelBase] = useState<'lite_mobilenet_v2' | 'mobilenet_v2' | 'mobilenet_v1'>('lite_mobilenet_v2');
  const [isModelLoading, setIsModelLoading] = useState(false);

  // Initialize Model
  useEffect(() => {
    async function init() {
      try {
        setIsModelLoading(true);
        await tf.ready();
        const model = await cocoSsd.load({ base: modelBase });
        modelRef.current = model;
        setIsLoading(false);
        setIsModelLoading(false);
      } catch (err) {
        console.error("Failed to load model:", err);
        setIsModelLoading(false);
      }
    }
    init();
  }, [modelBase]);

  // Intersection logic: Check if segment (a, b) crosses segment (c, d)
  const intersects = (a: Point, b: Point, c: Point, d: Point) => {
    const det = (b.x - a.x) * (d.y - c.y) - (d.x - c.x) * (b.y - a.y);
    if (det === 0) return false;
    const lambda = ((d.y - c.y) * (d.x - a.x) + (c.x - d.x) * (d.y - a.y)) / det;
    const gamma = ((a.y - b.y) * (d.x - a.x) + (b.x - a.x) * (d.y - a.y)) / det;
    return (0 < lambda && lambda < 1) && (0 < gamma && gamma < 1);
  };

  // Tracking and Counting Logic
  const processDetections = useCallback((detections: cocoSsd.DetectedObject[]) => {
    const now = Date.now();
    const currentDetections: Detection[] = detections
      .filter(d => d.score >= confidenceThreshold)
      .map(d => ({
        bbox: d.bbox,
        class: d.class,
        score: d.score,
        centroid: {
          x: d.bbox[0] + d.bbox[2] / 2,
          y: d.bbox[1] + d.bbox[3] / 2
        }
      }));

    const updatedTracked: TrackedObject[] = [];
    const usedDetectionIndices = new Set<number>();

    // Match existing tracked objects to new detections
    trackedObjectsRef.current.forEach(tracked => {
      let bestMatch = -1;
      let minDist = 50; // Max distance to associate

      currentDetections.forEach((det, idx) => {
        if (usedDetectionIndices.has(idx)) return;
        if (det.class !== tracked.class) return;

        const dist = Math.sqrt(
          Math.pow(det.centroid.x - tracked.lastCentroid.x, 2) +
          Math.pow(det.centroid.y - tracked.lastCentroid.y, 2)
        );

        if (dist < minDist) {
          minDist = dist;
          bestMatch = idx;
        }
      });

      if (bestMatch !== -1) {
        const det = currentDetections[bestMatch];
        usedDetectionIndices.add(bestMatch);

        // Check for line crossing
        if (!tracked.counted && line) {
          if (intersects(tracked.lastCentroid, det.centroid, line.p1, line.p2)) {
            let vehicleType: keyof CountData = 'other';
            if (det.class === 'car') vehicleType = 'car';
            else if (det.class === 'truck') vehicleType = 'truck';
            else if (det.class === 'bus') vehicleType = 'bus';
            else if (det.class === 'motorcycle') vehicleType = 'motorcycle';
            
            // Determine direction using cross product
            // (p.x - lineP1.x) * (lineP2.y - lineP1.y) - (p.y - lineP1.y) * (lineP2.x - lineP1.x)
            const sideBefore = (tracked.lastCentroid.x - line.p1.x) * (line.p2.y - line.p1.y) - (tracked.lastCentroid.y - line.p1.y) * (line.p2.x - line.p1.x);
            const direction: 'inbound' | 'outbound' = sideBefore > 0 ? 'inbound' : 'outbound';

            if (direction === 'inbound') {
              setInboundCounts(prev => ({ ...prev, [vehicleType]: prev[vehicleType] + 1 }));
            } else {
              setOutboundCounts(prev => ({ ...prev, [vehicleType]: prev[vehicleType] + 1 }));
            }

            setHistory(prev => [
              { timestamp: new Date().toLocaleTimeString(), type: vehicleType, id: tracked.id, direction },
              ...prev
            ]);
            tracked.counted = true;
          }
        }

        updatedTracked.push({
          ...tracked,
          lastCentroid: det.centroid,
          path: [...tracked.path.slice(-10), det.centroid],
          lastSeen: now
        });
      } else if (now - tracked.lastSeen < 1000) {
        // Keep for a bit even if not seen
        updatedTracked.push(tracked);
      }
    });

    // Add new detections as tracked objects
    currentDetections.forEach((det, idx) => {
      if (!usedDetectionIndices.has(idx)) {
        updatedTracked.push({
          id: nextIdRef.current++,
          class: det.class,
          lastCentroid: det.centroid,
          path: [det.centroid],
          lastSeen: now,
          counted: false
        });
      }
    });

    trackedObjectsRef.current = updatedTracked;
  }, [line, confidenceThreshold]);

  // Rendering Logic
  const render = useCallback(() => {
    if (!canvasRef.current || !videoRef.current) return;
    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return;

    const canvas = canvasRef.current;
    const video = videoRef.current;

    // Sync canvas size
    if (video.videoWidth > 0 && (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight)) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    if (canvas.width === 0 || canvas.height === 0) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw counting line
    if (line) {
      ctx.beginPath();
      ctx.moveTo(line.p1.x, line.p1.y);
      ctx.lineTo(line.p2.x, line.p2.y);
      ctx.strokeStyle = '#ef4444'; // Red for counting line
      ctx.lineWidth = 4;
      ctx.stroke();
      
      ctx.fillStyle = '#ef4444';
      ctx.beginPath();
      ctx.arc(line.p1.x, line.p1.y, 6, 0, Math.PI * 2);
      ctx.arc(line.p2.x, line.p2.y, 6, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = '#ef4444';
      ctx.font = 'bold 16px sans-serif';
      ctx.fillText('COUNTING LINE', line.p1.x, line.p1.y - 15);

      // Draw direction indicators
      const midX = (line.p1.x + line.p2.x) / 2;
      const midY = (line.p1.y + line.p2.y) / 2;
      const dx = line.p2.x - line.p1.x;
      const dy = line.p2.y - line.p1.y;
      const len = Math.sqrt(dx * dx + dy * dy);
      const nx = -dy / len;
      const ny = dx / len;

      ctx.font = '12px sans-serif';
      ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
      ctx.fillText('INBOUND', midX + nx * 30, midY + ny * 30);
      ctx.fillText('OUTBOUND', midX - nx * 30, midY - ny * 30);
    }

    // Draw drawing preview
    if (isDrawing && drawingStart) {
      // Start point indicator (always show once clicked)
      ctx.fillStyle = '#10b981';
      ctx.beginPath();
      ctx.arc(drawingStart.x, drawingStart.y, 6, 0, Math.PI * 2);
      ctx.fill();

      if (previewPoint) {
        ctx.beginPath();
        ctx.moveTo(drawingStart.x, drawingStart.y);
        ctx.lineTo(previewPoint.x, previewPoint.y);
        ctx.strokeStyle = '#10b981'; // Solid green for active drawing
        ctx.lineWidth = 3;
        ctx.setLineDash([5, 5]);
        ctx.stroke();
        ctx.setLineDash([]);

        // Preview end point
        ctx.fillStyle = 'rgba(16, 185, 129, 0.5)';
        ctx.beginPath();
        ctx.arc(previewPoint.x, previewPoint.y, 4, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Draw detections
    lastDetectionsRef.current.forEach(det => {
      const isVehicle = VEHICLE_CLASSES.includes(det.class);
      
      if (!debugMode && det.score < confidenceThreshold) return;
      if (debugMode && det.score < 0.1) return;
      
      const [x, y, w, h] = det.bbox;
      const isBelowThreshold = det.score < confidenceThreshold;
      
      if (isVehicle) {
        ctx.strokeStyle = isBelowThreshold ? '#94a3b8' : '#10b981';
      } else {
        ctx.strokeStyle = isBelowThreshold ? '#94a3b8' : '#3b82f6';
      }
      
      ctx.lineWidth = isBelowThreshold ? 1 : 2;
      ctx.setLineDash(isBelowThreshold ? [5, 5] : []);
      ctx.strokeRect(x, y, w, h);
      ctx.setLineDash([]);

      ctx.fillStyle = isBelowThreshold ? '#94a3b8' : (isVehicle ? '#10b981' : '#3b82f6');
      ctx.font = `${isBelowThreshold ? '10px' : '14px'} sans-serif`;
      ctx.fillText(`${det.class} (${Math.round(det.score * 100)}%)`, x, y > 20 ? y - 5 : y + 15);
    });
  }, [line, isDrawing, drawingStart, previewPoint, debugMode, confidenceThreshold]);

  // Main Loop
  const detectFrame = useCallback(async () => {
    if (!videoRef.current || !modelRef.current) return;
    if (!isPlaying) return;

    const detections = await modelRef.current.detect(videoRef.current);
    const mappedDetections = detections.map(d => ({
      ...d,
      class: (d.class === 'person' || d.class === 'bicycle') ? 'motorcycle' : d.class
    }));
    lastDetectionsRef.current = mappedDetections;
    processDetections(mappedDetections);
    render();

    if (isPlaying) {
      requestRef.current = requestAnimationFrame(() => detectFrame());
    }
  }, [isPlaying, processDetections, render]);

  useEffect(() => {
    if (isPlaying) {
      requestRef.current = requestAnimationFrame(() => detectFrame());
    } else if (requestRef.current) {
      cancelAnimationFrame(requestRef.current);
    }
    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, [isPlaying, detectFrame]);

  useEffect(() => {
    if (source && videoRef.current) {
      render();
    }
  }, [source, render]);

  // Handlers
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const url = URL.createObjectURL(file);
      if (videoRef.current) {
        videoRef.current.src = url;
        setSource('file');
        setIsPlaying(false); // Don't start playing immediately to give time to draw the line
      }
    }
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setSource('camera');
        setIsPlaying(true);
      }
    } catch (err) {
      console.error("Camera access denied:", err);
    }
  };

  const stopVideo = () => {
    setIsPlaying(false);
    if (videoRef.current) {
      const stream = videoRef.current.srcObject as MediaStream;
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
      videoRef.current.srcObject = null;
      videoRef.current.src = '';
    }
    setSource(null);
  };

  const handleCanvasDoubleClick = () => {
    setLine(null);
    setIsDrawing(false);
    setDrawingStart(null);
    setPreviewPoint(null);
    render();
  };

  const handleCanvasClick = (e: React.MouseEvent) => {
    if (!canvasRef.current || canvasRef.current.width === 0) return;
    const rect = canvasRef.current.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;
    
    const scaleX = canvasRef.current.width / rect.width;
    const scaleY = canvasRef.current.height / rect.height;
    const p = {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY
    };

    if (!isDrawing) {
      setLine(null);
      setDrawingStart(p);
      setPreviewPoint(p);
      setIsDrawing(true);
      render();
    } else if (drawingStart) {
      // Check if it's not a double click (very close points)
      const dist = Math.sqrt(Math.pow(p.x - drawingStart.x, 2) + Math.pow(p.y - drawingStart.y, 2));
      if (dist > 5) {
        setLine({ p1: drawingStart, p2: p });
        setIsDrawing(false);
        setDrawingStart(null);
        setPreviewPoint(null);
        render();
      }
    }
  };

  const handleCanvasMouseMove = (e: React.MouseEvent) => {
    if (!isDrawing || !canvasRef.current || canvasRef.current.width === 0) return;
    const rect = canvasRef.current.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;

    const scaleX = canvasRef.current.width / rect.width;
    const scaleY = canvasRef.current.height / rect.height;
    const p = {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY
    };
    setPreviewPoint(p);
    render();
  };

  const exportCSV = () => {
    const headers = ['Timestamp', 'Vehicle Type', 'ID', 'Direction'];
    const rows = history.map(h => [h.timestamp, h.type, h.id, h.direction]);
    const csvContent = [headers, ...rows].map(e => e.join(",")).join("\n");
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", `vehicle_counts_${new Date().toISOString()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const resetCounts = () => {
    setInboundCounts({ car: 0, truck: 0, bus: 0, motorcycle: 0, other: 0 });
    setOutboundCounts({ car: 0, truck: 0, bus: 0, motorcycle: 0, other: 0 });
    setHistory([]);
    trackedObjectsRef.current = [];
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-[#0a0a0a] flex flex-col items-center justify-center text-white font-sans">
        <div className="w-16 h-16 border-4 border-emerald-500 border-t-transparent rounded-full animate-spin mb-4"></div>
        <h1 className="text-2xl font-bold tracking-tight">Initializing AI Engine...</h1>
        <p className="text-zinc-400 mt-2">Loading {modelBase.replace(/_/g, ' ').toUpperCase()}</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-zinc-100 font-sans selection:bg-emerald-500/30 flex flex-col overflow-hidden">
      {/* Header */}
      <header className="border-b border-zinc-800 bg-zinc-900/50 backdrop-blur-md z-50 flex-shrink-0">
        <div className="max-w-7xl mx-auto px-4 h-12 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="p-1.5 bg-emerald-500/10 rounded-lg">
              <Car className="w-4 h-4 text-emerald-500" />
            </div>
            <div>
              <h1 className="font-bold text-sm tracking-tight">Vehicle Counter AI</h1>
              <p className="text-[8px] text-zinc-500 uppercase tracking-widest font-semibold">YOLO Real-time Detection</p>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            <button 
              onClick={() => setShowSettings(!showSettings)}
              className="p-1.5 hover:bg-zinc-800 rounded-full transition-colors"
            >
              <Settings className="w-4 h-4 text-zinc-400" />
            </button>
            <button 
              onClick={exportCSV}
              disabled={history.length === 0}
              className="flex items-center gap-2 px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white rounded-lg text-xs font-medium transition-all"
            >
              <Download className="w-3 h-3" />
              Export
            </button>
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-7xl mx-auto w-full p-4 grid grid-cols-1 lg:grid-cols-12 gap-4 overflow-hidden">
        {/* Left Column: Video & Controls */}
        <div className="lg:col-span-9 flex flex-col gap-4 overflow-hidden">
          {/* Video Container */}
          <div className="relative flex-1 bg-zinc-900 rounded-2xl overflow-hidden border border-zinc-800 shadow-2xl group min-h-0">
            {!source && (
              <div className="absolute inset-0 flex flex-col items-center justify-center p-4 text-center">
                <div className="w-16 h-16 bg-zinc-800 rounded-full flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-500">
                  <Play className="w-6 h-6 text-zinc-500" />
                </div>
                <h2 className="text-lg font-semibold mb-1">Select Video Source</h2>
                <p className="text-zinc-500 text-xs max-w-xs mb-6">
                  Upload a video file, use your webcam, or connect to a stream to start counting.
                </p>
                <div className="flex flex-wrap justify-center gap-2">
                  {!showStreamInput ? (
                    <>
                      <button 
                        onClick={startCamera}
                        className="flex items-center gap-2 px-4 py-2 bg-white text-black rounded-xl text-sm font-bold hover:bg-zinc-200 transition-all"
                      >
                        <Camera className="w-4 h-4" />
                        Webcam
                      </button>
                      <label className="flex items-center gap-2 px-4 py-2 bg-zinc-800 text-white rounded-xl text-sm font-bold hover:bg-zinc-700 cursor-pointer transition-all">
                        <Upload className="w-4 h-4" />
                        Upload
                        <input type="file" accept="video/*" className="hidden" onChange={handleFileUpload} />
                      </label>
                      <button 
                        onClick={() => setShowStreamInput(true)}
                        className="flex items-center gap-2 px-4 py-2 bg-zinc-800 text-white rounded-xl text-sm font-bold hover:bg-zinc-700 transition-all"
                      >
                        <Play className="w-4 h-4" />
                        Stream
                      </button>
                    </>
                  ) : (
                    <div className="w-full max-w-md animate-in zoom-in-95 duration-200">
                      <div className="flex gap-2">
                        <input 
                          type="text"
                          placeholder="Video URL..."
                          value={streamUrl}
                          onChange={(e) => setStreamUrl(e.target.value)}
                          className="flex-1 bg-zinc-800 border border-zinc-700 rounded-xl px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-emerald-500 transition-all"
                        />
                        <button 
                          onClick={() => {
                            if (streamUrl && videoRef.current) {
                              videoRef.current.src = streamUrl;
                              setSource('stream');
                              setIsPlaying(true);
                              setShowStreamInput(false);
                            }
                          }}
                          className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-xl text-sm font-bold transition-all"
                        >
                          Start
                        </button>
                        <button 
                          onClick={() => setShowStreamInput(false)}
                          className="p-2 bg-zinc-800 hover:bg-zinc-700 text-white rounded-xl transition-all"
                        >
                          <X className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            <video
              ref={videoRef}
              className={cn("w-full h-full object-contain", !source && "hidden")}
              autoPlay
              muted
              loop
              playsInline
              crossOrigin="anonymous"
              onLoadedData={() => detectFrame(true)}
              onPlay={() => {
                setIsPlaying(true);
                detectFrame(true);
              }}
              onError={() => {
                alert("Failed to load video stream.");
                setSource(null);
                setIsPlaying(false);
              }}
            />
            <canvas
              ref={canvasRef}
              onClick={handleCanvasClick}
              onDoubleClick={handleCanvasDoubleClick}
              onMouseMove={handleCanvasMouseMove}
              className={cn(
                "absolute inset-0 w-full h-full cursor-crosshair",
                !source && "hidden"
              )}
            />

            {/* Overlay Controls */}
            {source && (
              <div className="absolute bottom-4 left-4 right-4 flex items-center justify-between opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                <div className="flex items-center gap-2 bg-black/60 backdrop-blur-md p-1 rounded-lg border border-white/10">
                  <button 
                    onClick={() => setIsPlaying(!isPlaying)}
                    className="p-1.5 hover:bg-white/10 rounded-md transition-colors"
                  >
                    {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                  </button>
                  <button 
                    onClick={stopVideo}
                    className="p-1.5 hover:bg-red-500/20 text-red-400 rounded-md transition-colors"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
                
                <div className="bg-black/60 backdrop-blur-md px-2 py-1 rounded-lg border border-white/10 text-[10px] font-mono text-emerald-400 uppercase tracking-wider">
                  Live Detection
                </div>
              </div>
            )}
          </div>

          {/* Instructions - Compact */}
          <div className="bg-zinc-900/50 border border-zinc-800 rounded-xl p-3 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Plus className="w-4 h-4 text-emerald-500" />
              <p className="text-xs text-zinc-400">
                Click and drag on the video to draw a <span className="text-red-400 font-bold">Counting Line</span>.
              </p>
            </div>
            {line && (
              <button 
                onClick={() => { setLine(null); }}
                className="flex items-center gap-2 text-[10px] font-bold text-red-400 hover:text-red-300 transition-colors"
              >
                <Trash2 className="w-3 h-3" />
                Clear Line
              </button>
            )}
          </div>
        </div>

        {/* Right Column: Stats */}
        <div className="lg:col-span-3 flex flex-col gap-3 overflow-y-auto custom-scrollbar pr-1">
          {/* Total Counter - Compact Side-by-Side */}
          <div className="grid grid-cols-2 gap-2">
            <div className="bg-emerald-600 rounded-xl p-2 text-white shadow-lg shadow-emerald-900/20">
              <div className="flex items-center justify-between mb-0.5">
                <span className="text-emerald-100 text-[7px] font-bold uppercase tracking-wider">Total In</span>
                <ArrowDownLeft className="w-2.5 h-2.5 opacity-50" />
              </div>
              <div className="text-xl font-black leading-none">
                {(Object.values(inboundCounts) as number[]).reduce((a, b) => a + b, 0)}
              </div>
            </div>
            <div className="bg-blue-600 rounded-xl p-2 text-white shadow-lg shadow-blue-900/20">
              <div className="flex items-center justify-between mb-0.5">
                <span className="text-blue-100 text-[7px] font-bold uppercase tracking-wider">Total Out</span>
                <ArrowUpRight className="w-2.5 h-2.5 opacity-50" />
              </div>
              <div className="text-xl font-black leading-none">
                {(Object.values(outboundCounts) as number[]).reduce((a, b) => a + b, 0)}
              </div>
            </div>
          </div>

          {/* Stats Grid - Side-by-Side Columns */}
          <div className="grid grid-cols-2 gap-2">
            {/* Inbound Column */}
            <div className="space-y-2">
              <div className="flex items-center gap-1.5 px-1">
                <div className="w-1 h-1 rounded-full bg-emerald-500" />
                <h3 className="text-[9px] font-bold uppercase tracking-widest text-zinc-500">Inbound</h3>
              </div>
              <div className="grid grid-cols-1 gap-1">
                <StatCard icon={<Car />} label="Cars" value={inboundCounts.car} color="emerald" />
                <StatCard icon={<Truck />} label="Trucks" value={inboundCounts.truck} color="blue" />
                <StatCard icon={<Bus />} label="Buses" value={inboundCounts.bus} color="amber" />
                <StatCard icon={<Bike />} label="Motorcycles" value={inboundCounts.motorcycle} color="purple" />
              </div>
            </div>

            {/* Outbound Column */}
            <div className="space-y-2">
              <div className="flex items-center gap-1.5 px-1">
                <div className="w-1 h-1 rounded-full bg-blue-500" />
                <h3 className="text-[9px] font-bold uppercase tracking-widest text-zinc-500">Outbound</h3>
              </div>
              <div className="grid grid-cols-1 gap-1">
                <StatCard icon={<Car />} label="Cars" value={outboundCounts.car} color="emerald" />
                <StatCard icon={<Truck />} label="Trucks" value={outboundCounts.truck} color="blue" />
                <StatCard icon={<Bus />} label="Buses" value={outboundCounts.bus} color="amber" />
                <StatCard icon={<Bike />} label="Motorcycles" value={outboundCounts.motorcycle} color="purple" />
              </div>
            </div>
          </div>

          <button 
            onClick={resetCounts}
            className="mt-auto w-full py-1.5 bg-zinc-800 hover:bg-zinc-700 text-zinc-500 hover:text-red-400 rounded-lg text-[8px] font-bold uppercase tracking-widest transition-all flex items-center justify-center gap-1.5"
          >
            <Trash2 className="w-2.5 h-2.5" />
            Reset
          </button>
        </div>
      </main>

      {/* Settings Modal */}
      {showSettings && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-200">
          <div className="bg-zinc-900 border border-zinc-800 rounded-3xl w-full max-w-md overflow-hidden shadow-2xl">
            <div className="p-6 border-b border-zinc-800 flex items-center justify-between">
              <h2 className="text-xl font-bold">Detection Settings</h2>
              <button onClick={() => setShowSettings(false)} className="p-2 hover:bg-zinc-800 rounded-full">
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="p-6 space-y-6">
              {/* Model Selection */}
              <div className="space-y-3">
                <label className="text-sm font-medium text-zinc-400 flex items-center gap-2">
                  <Settings className="w-4 h-4" />
                  Detection Model
                </label>
                <div className="grid grid-cols-1 gap-2">
                  {[
                    { id: 'lite_mobilenet_v2', name: 'Lite MobileNet V2', desc: 'Fastest, optimized for mobile/web' },
                    { id: 'mobilenet_v2', name: 'MobileNet V2', desc: 'Balanced speed and accuracy' },
                    { id: 'mobilenet_v1', name: 'MobileNet V1', desc: 'Legacy model' }
                  ].map((m) => (
                    <button
                      key={m.id}
                      onClick={() => setModelBase(m.id as any)}
                      disabled={isModelLoading}
                      className={cn(
                        "flex flex-col items-start p-3 rounded-xl border transition-all text-left",
                        modelBase === m.id 
                          ? "bg-emerald-500/10 border-emerald-500 text-emerald-500" 
                          : "bg-zinc-800 border-zinc-700 text-zinc-400 hover:border-zinc-600"
                      )}
                    >
                      <div className="flex items-center justify-between w-full">
                        <span className="font-bold text-sm">{m.name}</span>
                        {modelBase === m.id && isModelLoading && (
                          <div className="w-3 h-3 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
                        )}
                      </div>
                      <span className="text-[10px] opacity-60 mt-0.5">{m.desc}</span>
                    </button>
                  ))}
                </div>
              </div>

              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <label className="text-sm font-medium text-zinc-400">Debug Mode</label>
                  <button 
                    onClick={() => setDebugMode(!debugMode)}
                    className={cn(
                      "w-12 h-6 rounded-full transition-all relative",
                      debugMode ? "bg-emerald-500" : "bg-zinc-700"
                    )}
                  >
                    <div className={cn(
                      "absolute top-1 w-4 h-4 bg-white rounded-full transition-all",
                      debugMode ? "left-7" : "left-1"
                    )} />
                  </button>
                </div>
                <p className="text-[10px] text-zinc-500">
                  Shows all detected objects (even below threshold) with dashed lines to help tune detection.
                </p>
              </div>

              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <label className="text-sm font-medium text-zinc-400">Confidence Threshold</label>
                  <span className="text-emerald-500 font-mono font-bold">{Math.round(confidenceThreshold * 100)}%</span>
                </div>
                <input 
                  type="range" 
                  min="0.1" 
                  max="0.9" 
                  step="0.05" 
                  value={confidenceThreshold}
                  onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                  className="w-full h-2 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                />
                <p className="text-[10px] text-zinc-500">
                  Higher threshold reduces false positives but might miss some vehicles.
                </p>
              </div>

              <div className="pt-4 border-t border-zinc-800">
                <button 
                  onClick={() => {
                    setLine(null);
                    setShowSettings(false);
                  }}
                  className="w-full py-3 bg-zinc-800 hover:bg-zinc-700 text-white rounded-xl font-bold transition-all"
                >
                  Clear Counting Line
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      <style dangerouslySetInnerHTML={{ __html: `
        .custom-scrollbar::-webkit-scrollbar { width: 4px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: #27272a; border-radius: 10px; }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: #3f3f46; }
      `}} />
    </div>
  );
}

function StatCard({ icon, label, value, color }: { icon: React.ReactNode, label: string, value: number, color: 'emerald' | 'blue' | 'amber' | 'purple' }) {
  const colors = {
    emerald: 'bg-emerald-500/10 text-emerald-500 border-emerald-500/20',
    blue: 'bg-blue-500/10 text-blue-500 border-blue-500/20',
    amber: 'bg-amber-500/10 text-amber-500 border-amber-500/20',
    purple: 'bg-purple-500/10 text-purple-500 border-purple-500/20',
  };

  return (
    <div className={cn("p-2 px-3 rounded-xl border bg-zinc-900 flex items-center gap-3 transition-all hover:bg-zinc-800", colors[color])}>
      <div className="opacity-80 scale-75">{icon}</div>
      <div className="flex flex-col items-start">
        <div className="text-lg font-black leading-none">{value}</div>
        <div className="text-[8px] font-bold uppercase tracking-widest opacity-60">{label}</div>
      </div>
    </div>
  );
}
