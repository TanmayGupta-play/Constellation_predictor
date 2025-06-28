import cv2
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse, JsonResponse
import numpy as np
from ultralytics import YOLO
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
import base64
from datetime import datetime
import os
from django.conf import settings
import threading
import queue
import time
from PIL import Image
import io

# Global models - load once at startup
COCO_MODEL = None
CONSTELLATION_MODEL = None

# Frame processing queue for optimization
frame_queue = queue.Queue(maxsize=2)  # Limit queue size to prevent memory buildup
processing_lock = threading.Lock()

# Performance tracking
last_process_time = 0
min_process_interval = 0.1  # Minimum 100ms between processes

def initialize_models():
    """Initialize models once at startup"""
    global COCO_MODEL, CONSTELLATION_MODEL
    
    if COCO_MODEL is None:
        try:
            COCO_MODEL = YOLO('yolov8n.pt')
            print("COCO model loaded successfully")
        except Exception as e:
            print(f"Error loading COCO model: {e}")
    
    if CONSTELLATION_MODEL is None:
        try:
            CONSTELLATION_MODEL = YOLO(r'.\Predictor\REAL_TIME_DETECTOR\runs\detect\train\weights\best.pt')
            print("Constellation model loaded successfully")
        except Exception as e:
            print(f"Error loading Constellation model: {e}")

# Initialize models on module load
initialize_models()

# Constellation class names (update according to your model)
CONSTELLATION_CLASSES = {
    0: "Andromeda", 1: "Antlia", 2: "Apus", 3: "Aquarius", 4: "Aquila",
    5: "Ara", 6: "Aries", 7: "Auriga", 8: "Bootes", 9: "Caelum",
    10: "Camelopardalis", 11: "Cancer", 12: "Canes Venatici", 13: "Canis Major", 14: "Canis Minor",
    15: "Capricornus", 16: "Carina", 17: "Cassiopeia", 18: "Centaurus", 19: "Cepheus",
    20: "Cetus", 21: "Chamaeleon", 22: "Circinus", 23: "Columba", 24: "Coma Berenices",
    25: "Corona Australis", 26: "Corona Borealis", 27: "Corvus", 28: "Crater", 29: "Crux",
    30: "Cygnus", 31: "Delphinus", 32: "Dorado", 33: "Draco", 34: "Equuleus",
    35: "Eridanus", 36: "Fornax", 37: "Gemini", 38: "Grus", 39: "Hercules",
    40: "Horologium", 41: "Hydra", 42: "Hydrus", 43: "Indus", 44: "Lacerta",
    45: "Leo", 46: "Leo Minor", 47: "Lepus", 48: "Libra", 49: "Lupus",
    50: "Lynx", 51: "Lyra", 52: "Mensa", 53: "Microscopium", 54: "Monoceros",
    55: "Musca", 56: "Norma", 57: "Octans", 58: "Ophiuchus", 59: "Orion",
    60: "Pavo", 61: "Pegasus", 62: "Perseus", 63: "Phoenix", 64: "Pictor",
    65: "Pisces", 66: "Piscis Austrinus", 67: "Puppis", 68: "Pyxis", 69: "Reticulum",
    70: "Sagitta", 71: "Sagittarius", 72: "Scorpius", 73: "Sculptor", 74: "Scutum",
    75: "Serpens", 76: "Sextans", 77: "Taurus", 78: "Telescopium", 79: "Triangulum",
    80: "Triangulum Australe", 81: "Tucana", 82: "Ursa Major", 83: "Ursa Minor", 84: "Vela",
    85: "Virgo", 86: "Volans", 87: "Vulpecula"
}

def resize_frame_for_processing(frame, max_size=640):
    """Resize frame to reduce processing time while maintaining aspect ratio"""
    height, width = frame.shape[:2]
    if max(height, width) > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        return resized, (width / new_width, height / new_height)
    return frame, (1.0, 1.0)

def fast_detect_objects(frame, model, confidence_threshold=0.3):
    """Optimized object detection with reduced confidence threshold for speed"""
    try:
        # Run inference with optimized parameters
        results = model(
            frame, 
            verbose=False,
            conf=confidence_threshold,  # Lower threshold for faster processing
            iou=0.7,  # Higher IoU threshold to reduce overlapping boxes
            max_det=50,  # Limit maximum detections
            half=False,  # Disable half precision for stability
            device='cpu'  # Explicitly use CPU (change to 'cuda' if GPU available)
        )
        return results
    except Exception as e:
        print(f"Detection error: {e}")
        return None

@csrf_exempt
def process_frame(request):
    """Optimized frame processing for real-time video"""
    global last_process_time
    
    if request.method != 'POST' or 'frame' not in request.FILES:
        return HttpResponse(status=400)
    
    # Rate limiting - skip frames if processing too frequently
    current_time = time.time()
    if current_time - last_process_time < min_process_interval:
        # Return previous frame or skip processing
        return HttpResponse(b'', status=204)  # No content
    
    try:
        # Quick validation
        file = request.FILES['frame']
        if file.size > 5 * 1024 * 1024:  # 5MB limit for video frames
            return HttpResponse(status=413)  # Payload too large
        
        # Fast image decoding
        img_data = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        
        if frame is None:
            return HttpResponse(status=400)
        
        # Resize frame for faster processing
        processed_frame, scale_factors = resize_frame_for_processing(frame, max_size=416)
        
        # Initialize variables
        max_confidence = 0
        annotated_frame = processed_frame.copy()
        use_constellation_model = True
        
        # Quick COCO detection with lower confidence threshold
        if COCO_MODEL is not None:
            coco_results = fast_detect_objects(processed_frame, COCO_MODEL, confidence_threshold=0.4)
            
            if coco_results and len(coco_results) > 0:
                # Check for high-confidence detections
                for result in coco_results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            conf = float(box.conf[0])
                            if conf > max_confidence:
                                max_confidence = conf
                
                # Use COCO results if high confidence object detected
                if max_confidence >= 0.75:  # Lowered threshold for better responsiveness
                    try:
                        use_constellation_model = False
                    except:
                        # Fallback to manual annotation if plot fails
                        pass
        
        # Use constellation model if no high-confidence objects detected
        if use_constellation_model and CONSTELLATION_MODEL is not None:
            constellation_results = fast_detect_objects(processed_frame, CONSTELLATION_MODEL, confidence_threshold=0.3)
            
            if constellation_results and len(constellation_results) > 0:
                try:
                    annotated_frame = constellation_results[0].plot()
                except:
                    # Fallback - just use original frame
                    annotated_frame = processed_frame
        
        # Scale back up if frame was resized
        if scale_factors != (1.0, 1.0):
            original_size = (frame.shape[1], frame.shape[0])  # (width, height)
            annotated_frame = cv2.resize(annotated_frame, original_size, interpolation=cv2.INTER_LINEAR)
        
        # Add simple performance info
        processing_time = time.time() - current_time
        fps = 1.0 / max(processing_time, 0.001)
        
        cv2.putText(annotated_frame, f"FPS: {fps:.1f} | Conf: {max_confidence:.2f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Fast JPEG encoding with lower quality for speed
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 75]  # Lower quality for speed
        ret, buffer = cv2.imencode('.jpg', annotated_frame, encode_params)
        
        if not ret:
            return HttpResponse(status=500)
        
        # Update timing
        last_process_time = current_time
        
        return HttpResponse(buffer.tobytes(), content_type='image/jpeg')
        
    except Exception as e:
        print(f"Frame processing error: {e}")
        return HttpResponse(status=500)

def home(request):
    return render(request, 'home.html')

def detect_view(request):
    return render(request, 'detect.html')

def database(request):
    return render(request,'database.html')

def upload(request):
    return render(request,'upload.html')

def predict(request):
    return render(request, 'predict.html')

@csrf_exempt
def process_upload(request):
    """Original upload processing - keeping full quality for static images"""
    if request.method == 'POST':
        try:
            # Check if image file is uploaded
            if 'image' not in request.FILES:
                messages.error(request, 'No image file uploaded.')
                return redirect('upload')
            
            file = request.FILES['image']
            location = request.POST.get('location', '').strip()
            capture_time = request.POST.get('capture_time', '')
            
            # Validate file type
            allowed_extensions = ['.jpg', '.jpeg', '.png']
            file_extension = os.path.splitext(file.name)[1].lower()
            if file_extension not in allowed_extensions:
                messages.error(request, 'Invalid file format. Please upload JPG, JPEG, or PNG files only.')
                return redirect('upload')
            
            # Validate file size (10MB limit)
            max_size = 10 * 1024 * 1024  # 10MB
            if file.size > max_size:
                messages.error(request, 'File size too large. Please upload files smaller than 10MB.')
                return redirect('upload')
            
            # Read and decode image
            img_data = np.frombuffer(file.read(), np.uint8)
            frame = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            
            if frame is None:
                messages.error(request, 'Invalid image file. Please upload a valid image.')
                return redirect('upload')
            
            height, width = frame.shape[:2]
            
            coco_results = COCO_MODEL(frame, verbose=False) if COCO_MODEL else None
            max_confidence = 0
            detected_objects = []
            
            if coco_results:
                for result in coco_results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            conf = float(box.conf[0])
                            if conf > max_confidence:
                                max_confidence = conf
                            
                            # Get class name
                            class_id = int(box.cls[0])
                            class_name = COCO_MODEL.names[class_id]
                            detected_objects.append({
                                'name': class_name,
                                'confidence': conf
                            })
            
            use_constellation_model = max_confidence < 0.87
            detected_constellations = []
            annotated_frame = None
            analysis_type = ""
            
            if use_constellation_model and CONSTELLATION_MODEL is not None:
                constellation_results = CONSTELLATION_MODEL(frame, verbose=False)
                annotated_frame = constellation_results[0].plot()
                analysis_type = "constellation"
                
                for result in constellation_results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            conf = float(box.conf[0])
                            class_id = int(box.cls[0])
                            
                            constellation_name = CONSTELLATION_CLASSES.get(class_id, f"Unknown_{class_id}")
                            
                            detected_constellations.append({
                                'name': constellation_name,
                                'confidence': conf,
                                'coordinates': box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                            })
                
                print(f"Detected constellations: {detected_constellations}")
            else:
                if coco_results:
                    pass
                else:
                    annotated_frame = frame
                analysis_type = "objects"
                print(f"Detected objects: {detected_objects}")
            
            info_text = f"Analysis: {analysis_type.title()} | Max Conf: {max_confidence:.2f}"
            if location:
                info_text += f" | Location: {location}"
            
            cv2.putText(annotated_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                messages.error(request, 'Error processing image.')
                return redirect('upload')
            
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            context = {
                'processed_image': img_base64,
                'original_filename': file.name,
                'file_size': f"{file.size / 1024 / 1024:.2f} MB",
                'image_dimensions': f"{width} x {height}",
                'location': location,
                'capture_time': capture_time,
                'analysis_type': analysis_type,
                'max_confidence': max_confidence,
                'detected_constellations': detected_constellations,
                'detected_objects': detected_objects,
                'total_detections': len(detected_constellations) if use_constellation_model else len(detected_objects),
                'processing_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return render(request, 'results.html', context)
            
        except Exception as e:
            print(f"Error processing upload: {str(e)}")
            messages.error(request, f'Error processing image: {str(e)}')
            return redirect('upload')
    
    return redirect('upload')

def chatbot(request):
    return render(request, 'chatbot.html')

# cloudflared tunnel --url http://localhost:8000