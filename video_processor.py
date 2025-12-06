"""
Multi-threaded video capture and processing
"""
import cv2
import threading
import queue
import time
from collections import deque
import numpy as np

class ThreadedVideoCapture:
    """
    Threaded video capture for better performance
    """
    def __init__(self, src=0):
        """Initialize video capture in separate thread"""
        self.cap = cv2.VideoCapture(src)
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Frame queue
        self.q = queue.Queue(maxsize=10)
        self.stopped = False
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Start capture thread
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
    
    def _reader(self):
        """Read frames in background thread"""
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                break
            
            # Add to queue, remove oldest if full
            if not self.q.full():
                self.q.put(frame)
            else:
                try:
                    self.q.get_nowait()
                    self.q.put(frame)
                except queue.Empty:
                    pass
            
            self.frame_count += 1
    
    def read(self):
        """Get latest frame from queue"""
        try:
            frame = self.q.get(timeout=1.0)
            
            # Calculate FPS
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                self.fps = self.frame_count / elapsed
            
            return True, frame
        except queue.Empty:
            return False, None
    
    def get_fps(self):
        """Get current FPS"""
        return self.fps
    
    def release(self):
        """Stop capture and release resources"""
        self.stopped = True
        self.thread.join()
        self.cap.release()
    
    def isOpened(self):
        """Check if capture is open"""
        return self.cap.isOpened() and not self.stopped


class FrameProcessor:
    """
    Process frames asynchronously
    """
    def __init__(self, captioner, classifier, alert_manager, process_every_n=3):
        """Initialize frame processor"""
        self.captioner = captioner
        self.classifier = classifier
        self.alert_manager = alert_manager
        self.process_every_n = process_every_n
        
        # Processing queue
        self.input_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=5)
        
        # Control
        self.stopped = False
        self.frame_counter = 0
        
        # Start processing thread
        self.thread = threading.Thread(target=self._processor, daemon=True)
        self.thread.start()
    
    def _processor(self):
        """Process frames in background thread"""
        while not self.stopped:
            try:
                frame_data = self.input_queue.get(timeout=0.1)
                
                frame = frame_data['frame']
                frame_num = frame_data['frame_num']
                
                # Generate caption
                caption = self.captioner.generate_caption(frame)
                
                # Classify caption
                prediction = self.classifier.classify(caption)
                
                # Check for alerts
                alert_triggered = self.alert_manager.should_trigger_alert(
                    prediction['label'],
                    prediction['confidence']
                )
                
                if alert_triggered:
                    self.alert_manager.log_alert(
                        prediction['label'],
                        prediction['confidence'],
                        caption,
                        frame
                    )
                
                # Put result
                result = {
                    'frame_num': frame_num,
                    'caption': caption,
                    'prediction': prediction,
                    'alert_triggered': alert_triggered
                }
                
                self.result_queue.put(result)
                
            except queue.Empty:
                continue
    
    def process_frame(self, frame):
        """
        Add frame to processing queue
        
        Args:
            frame: numpy array
            
        Returns:
            bool: whether frame was queued
        """
        self.frame_counter += 1
        
        # Skip frames for performance
        if self.frame_counter % self.process_every_n != 0:
            return False
        
        # Add to queue if not full
        if not self.input_queue.full():
            self.input_queue.put({
                'frame': frame.copy(),
                'frame_num': self.frame_counter
            })
            return True
        
        return False
    
    def get_result(self):
        """Get latest processing result"""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def stop(self):
        """Stop processor"""
        self.stopped = True
        self.thread.join()
