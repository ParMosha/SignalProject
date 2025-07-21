import cv2
import numpy as np
import time
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. YOLOv11 features will be disabled.")
    print("Install with: pip install ultralytics")

class ObjectTracker:
    """
    Implementation of an adaptive object tracking algorithm based on boosted Haar-like features
    and online discriminative learning. The tracker employs a bag-of-patches approach where
    multiple patches around the target location form positive training instances, while distant
    patches serve as negative examples. AdaBoost is used to combine weak Haar-like feature
    classifiers into a strong discriminative model that can distinguish target from background.
    
    Key Technical Components:
    1. Haar-like Feature Extraction: Two-rectangle features computed via integral images
    2. AdaBoost Learning: Weighted combination of weak decision stump classifiers
    3. Online Update Mechanism: Incremental learning to adapt to appearance changes
    4. Patch-based Search: Exhaustive search in local neighborhood for target localization
    """

    def __init__(self, n_features=250, n_samples=200, search_radius=20, learning_rate=0.85):
        """
        Initializes the adaptive object tracker with specified hyperparameters.

        Args:
            n_features (int): Number of Haar-like weak classifiers in the boosted ensemble
            n_samples (int): Number of negative training samples for discriminative learning
            search_radius (int): Spatial search radius (pixels) around previous target location
            learning_rate (float): Exponential smoothing factor for online classifier updates [0,1]
        """
        self.n_features = n_features
        self.n_samples = n_samples
        self.search_radius = search_radius
        self.learning_rate = learning_rate

        # Learned model components
        self.features = []          # List of (rect1, rect2) Haar-like feature descriptors
        self.classifiers = []       # Ensemble of weak classifiers with weights and thresholds
        self.bbox = None           # Current target bounding box (x, y, w, h)
        self.roi_size = (0, 0)     # Normalized ROI dimensions for feature computation


class YOLOTracker:
    """
    Hybrid tracking architecture combining deep learning object detection (YOLOv11) with
    online discriminative tracking. This approach leverages periodic YOLO detections to
    reinitialize and correct drift in the discriminative tracker, creating a robust
    multi-modal tracking system suitable for both single and multi-object scenarios.
    
    Technical Architecture:
    1. Primary Tracking: Online discriminative learning with Haar features and AdaBoost
    2. Detection Module: YOLOv11 CNN for periodic object detection and classification
    3. Association Logic: Hungarian-style matching between detections and active trackers
    4. Multi-Object Support: Independent tracker instances with unique identifiers
    """

    def __init__(self, n_features=250, n_samples=200, search_radius=20, learning_rate=0.85, detection_interval=10, track_everything=False):
        """
        Initializes the hybrid YOLO-enhanced tracking system.

        Args:
            n_features (int): Number of weak Haar classifiers per discriminative tracker
            n_samples (int): Negative sample count for online learning
            search_radius (int): Local search window radius in pixels
            learning_rate (float): Temporal smoothing factor for online adaptation
            detection_interval (int): YOLO detection frequency (every N frames)
            track_everything (bool): Multi-object mode vs single-object tracking
        """
        self.track_everything = track_everything
        if track_everything:
            self.trackers = {}  # Dictionary to store multiple trackers
            self.next_tracker_id = 0
        else:
            self.tracker = ObjectTracker(n_features, n_samples, search_radius, learning_rate)
        
        self.n_features = n_features
        self.n_samples = n_samples
        self.search_radius = search_radius
        self.learning_rate = learning_rate
        self.detection_interval = detection_interval
        self.frame_count = 0
        self.yolo_model = None
        self.target_class = None
        self.confidence_threshold = 0.5
        
        if YOLO_AVAILABLE:
            try:
                self.yolo_model = YOLO('yolo11s.pt')
                print("YOLOv11s model loaded successfully")
            except Exception as e:
                print(f"Failed to load YOLOv11s model: {e}")
                self.yolo_model = None

    def _detect_objects(self, frame):
        """
        Performs CNN-based object detection using YOLOv11 architecture.
        
        Technical Process:
        1. Forward pass through YOLOv11 backbone (CSPDarknet + FPN)
        2. Multi-scale prediction heads generate bounding box proposals
        3. Non-maximum suppression filters overlapping detections
        4. Confidence thresholding removes low-probability predictions
        
        Args:
            frame: Input BGR image tensor for detection
            
        Returns:
            List of detection dictionaries with bbox, confidence, class_id, class_name
        """
        if self.yolo_model is None:
            return []
        
        try:
            results = self.yolo_model(frame, verbose=False)
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        conf = float(box.conf)
                        if conf > self.confidence_threshold:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            class_id = int(box.cls)
                            class_name = self.yolo_model.names[class_id]
                            
                            # Convert to (x, y, w, h) format
                            bbox = (x1, y1, x2 - x1, y2 - y1)
                            detections.append({
                                'bbox': bbox,
                                'confidence': conf,
                                'class_id': class_id,
                                'class_name': class_name
                            })
            
            return detections
        except Exception as e:
            print(f"YOLO detection failed: {e}")
            return []

    def _select_target_from_detections(self, frame, detections):
        """Allow user to select which detected object to track."""
        if not detections:
            return None
        
        if self.track_everything:
            # In track everything mode, return all detections
            return [det['bbox'] for det in detections]
        
        # Draw all detections
        display_frame = frame.copy()
        for i, det in enumerate(detections):
            x, y, w, h = det['bbox']
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(display_frame, f"{i}: {det['class_name']} ({det['confidence']:.2f})", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        cv2.putText(display_frame, "YOLO Detection - Press number key to select object", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(display_frame, "Press 's' to skip YOLO and continue tracking", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(display_frame, "Press 'a' to track all detected objects", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow('Object Tracker', display_frame)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                return None
            elif key == ord('s'):  # Skip YOLO detection
                return None
            elif key == ord('a'):  # Track all objects
                self.track_everything = True
                return [det['bbox'] for det in detections]
            elif key >= ord('0') and key <= ord('9'):
                selected_idx = key - ord('0')
                if 0 <= selected_idx < len(detections):
                    selected = detections[selected_idx]
                    self.target_class = selected['class_id']
                    return selected['bbox']
        
        return None

    def _find_best_detection_match(self, detections, current_bbox):
        """Find the best matching detection based on current tracking."""
        if not detections or current_bbox is None:
            return None
        
        cx, cy, cw, ch = current_bbox
        current_center = (cx + cw//2, cy + ch//2)
        
        best_match = None
        best_distance = float('inf')
        
        for det in detections:
            # If we have a target class, prefer detections of the same class
            if self.target_class is not None and det['class_id'] != self.target_class:
                continue
            
            dx, dy, dw, dh = det['bbox']
            det_center = (dx + dw//2, dy + dh//2)
            
            # Calculate distance between centers
            distance = np.sqrt((current_center[0] - det_center[0])**2 + 
                             (current_center[1] - det_center[1])**2)
            
            if distance < best_distance:
                best_distance = distance
                best_match = det
        
        return best_match['bbox'] if best_match else None

    def _match_detections_to_trackers(self, detections):
        """
        Implements bipartite matching between YOLO detections and active trackers using
        Euclidean distance in image coordinates. This association step is critical for
        maintaining temporal consistency across frames in multi-object tracking.
        
        Technical Algorithm:
        1. Compute pairwise distances between tracker centers and detection centers
        2. Apply distance threshold to reject implausible associations
        3. Greedy assignment strategy favoring minimum distance matches
        4. Handle unmatched detections by spawning new tracker instances
        5. Mark unmatched trackers for potential removal
        
        Args:
            detections: List of YOLO detection results with bounding boxes
            
        Returns:
            tuple: (matches_dict, trackers_to_remove_list)
        """
        if not self.track_everything or not detections:
            return {}, []
        
        matches = {}
        used_detections = set()
        matched_trackers = set()
        
        # For each existing tracker, find the closest detection
        for tracker_id, tracker_info in self.trackers.items():
            current_bbox = tracker_info['tracker'].bbox
            if current_bbox is None:
                continue
                
            cx, cy, cw, ch = current_bbox
            current_center = (cx + cw//2, cy + ch//2)
            
            best_match_idx = None
            best_distance = float('inf')
            
            for i, det in enumerate(detections):
                if i in used_detections:
                    continue
                    
                dx, dy, dw, dh = det['bbox']
                det_center = (dx + dw//2, dy + dh//2)
                
                distance = np.sqrt((current_center[0] - det_center[0])**2 + 
                                 (current_center[1] - det_center[1])**2)
                
                # Only match if distance is reasonable (within search radius * 3)
                if distance < self.search_radius * 3 and distance < best_distance:
                    best_distance = distance
                    best_match_idx = i
            
            if best_match_idx is not None:
                matches[tracker_id] = detections[best_match_idx]['bbox']
                used_detections.add(best_match_idx)
                matched_trackers.add(tracker_id)
        
        # Find trackers that don't have matches (should be removed)
        trackers_to_remove = []
        for tracker_id in self.trackers.keys():
            if tracker_id not in matched_trackers:
                trackers_to_remove.append(tracker_id)
        
        # Create new trackers for unmatched detections
        for i, det in enumerate(detections):
            if i not in used_detections:
                new_tracker = ObjectTracker(self.n_features, self.n_samples, 
                                       self.search_radius, self.learning_rate)
                tracker_info = {
                    'tracker': new_tracker,
                    'class_id': det['class_id'],
                    'class_name': det['class_name']
                }
                matches[self.next_tracker_id] = det['bbox']
                self.trackers[self.next_tracker_id] = tracker_info
                self.next_tracker_id += 1
        
        return matches, trackers_to_remove

    def init(self, frame, bbox=None):
        """Initialize the tracker with optional YOLO detection."""
        if bbox is None and self.yolo_model is not None:
            # Use YOLO to detect objects and let user select
            detections = self._detect_objects(frame)
            if detections:
                bbox = self._select_target_from_detections(frame, detections)
            
            if bbox is None:
                # Fall back to manual selection
                cv2.putText(frame, "Select object to track, then press SPACE to continue", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imshow('Object Tracker', frame)
                
                print("Select the object you want to track in the video window, then press SPACE to start tracking")
                while True:
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC to exit
                        return False
                    elif key == ord(' '): 
                        bbox = cv2.selectROI('Object Tracker', frame, fromCenter=False, showCrosshair=True)
                        if bbox[2] > 0 and bbox[3] > 0:
                            break
                        else:
                            print("Invalid selection. Try again by pressing SPACE")
        
        if bbox is not None:
            if self.track_everything and isinstance(bbox, list):
                # Initialize multiple trackers
                self.trackers = {}
                self.next_tracker_id = 0
                for i, bb in enumerate(bbox):
                    if bb[2] > 0 and bb[3] > 0:
                        tracker = ObjectTracker(self.n_features, self.n_samples, 
                                           self.search_radius, self.learning_rate)
                        tracker.init(frame, bb)
                        self.trackers[i] = {
                            'tracker': tracker,
                            'class_id': None,
                            'class_name': 'Object'
                        }
                        self.next_tracker_id = i + 1
                self.frame_count = 0
                return len(self.trackers) > 0
            elif not self.track_everything and not isinstance(bbox, list):
                # Single tracker initialization
                if bbox[2] > 0 and bbox[3] > 0:
                    if not hasattr(self, 'tracker'):
                        self.tracker = ObjectTracker(self.n_features, self.n_samples, 
                                                    self.search_radius, self.learning_rate)
                    self.tracker.init(frame, bbox)
                    self.frame_count = 0
                    return True
        return False

    def update(self, frame):
        """Update tracking with periodic YOLO re-detection."""
        self.frame_count += 1
        
        if self.track_everything:
            # Multi-object tracking mode
            # Run YOLO detection every N frames
            if self.frame_count % self.detection_interval == 0 and self.yolo_model is not None:
                detections = self._detect_objects(frame)
                if detections:
                    matches, trackers_to_remove = self._match_detections_to_trackers(detections)
                    
                    # Remove trackers that don't have corresponding YOLO detections
                    for tracker_id in trackers_to_remove:
                        if tracker_id in self.trackers:
                            del self.trackers[tracker_id]
                            print(f"Removed tracker {tracker_id} - no longer detected by YOLO")
                    
                    # Update matched trackers with new detections
                    for tracker_id, new_bbox in matches.items():
                        if tracker_id in self.trackers:
                            self.trackers[tracker_id]['tracker'].init(frame, new_bbox)
                else:
                    # If no detections, remove all trackers after several failed detection cycles
                    if hasattr(self, '_no_detection_count'):
                        self._no_detection_count += 1
                    else:
                        self._no_detection_count = 1
                    
                    # Remove all trackers if no detections for 3 consecutive YOLO cycles
                    if self._no_detection_count >= 3:
                        print("No objects detected for multiple cycles, clearing all trackers")
                        self.trackers.clear()
                        self._no_detection_count = 0
                
                # Reset no detection counter if we had detections
                if detections:
                    self._no_detection_count = 0
            
            # Update all trackers
            results = {}
            trackers_to_remove = []
            
            for tracker_id, tracker_info in self.trackers.items():
                try:
                    new_bbox = tracker_info['tracker'].update(frame)
                    if new_bbox is not None:
                        results[tracker_id] = {
                            'bbox': new_bbox,
                            'class_name': tracker_info.get('class_name', 'Object'),
                            'class_id': tracker_info.get('class_id', None)
                        }
                    else:
                        # Mark for removal if tracking failed
                        trackers_to_remove.append(tracker_id)
                except Exception as e:
                    print(f"Tracker {tracker_id} failed: {e}")
                    trackers_to_remove.append(tracker_id)
            
            # Remove failed trackers
            for tracker_id in trackers_to_remove:
                del self.trackers[tracker_id]
            
            return results
        else:
            # Single object tracking mode
            # Run YOLO detection every N frames
            if self.frame_count % self.detection_interval == 0 and self.yolo_model is not None:
                detections = self._detect_objects(frame)
                if detections:
                    # Try to find a matching detection
                    current_bbox = self.tracker.bbox
                    new_bbox = self._find_best_detection_match(detections, current_bbox)
                    
                    if new_bbox is not None:
                        # Re-initialize tracker with YOLO detection
                        self.tracker.init(frame, new_bbox)
                        return new_bbox
            
            # Regular tracking
            return self.tracker.update(frame)

    @property
    def bbox(self):
        """Get current bounding box(es)."""
        if self.track_everything:
            return {tid: info['tracker'].bbox for tid, info in self.trackers.items()}
        else:
            return self.tracker.bbox


# Add ObjectTracker methods
def _get_integral_image(self, frame):
    """
    Computes integral image (summed area table) for O(1) rectangular region summation.
    This preprocessing step enables efficient Haar-like feature computation by converting
    arbitrary rectangle sums into four corner lookups.
    
    Mathematical Foundation:
    I(x,y) = Σ(i=0 to x, j=0 to y) image(i,j)
    Rectangle sum = I(x2,y2) + I(x1,y1) - I(x2,y1) - I(x1,y2)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.integral(gray)

def _generate_features(self, width, height):
    """
    Generates randomized Haar-like feature descriptors for discriminative learning.
    Each feature consists of two adjacent rectangles whose intensity difference
    captures local gradient and edge information robust to illumination changes.
    
    Feature Types:
    - Horizontal: Two vertically stacked rectangles (edge detection)
    - Vertical: Two horizontally adjacent rectangles (ridge detection)
    
    Randomization ensures diverse feature pool covering various scales and positions
    within the target region, following the Viola-Jones feature selection paradigm.
    """
    features = []
    for _ in range(self.n_features):
        # Two-rectangle feature (vertical or horizontal)
        x = np.random.randint(width)
        y = np.random.randint(height)
        w = np.random.randint(width - x)
        h = np.random.randint(height - y)
        
        # Feature is defined by (x, y, w, h, type), where type is horizontal/vertical
        # We'll represent this by the two rectangles
        if np.random.rand() > 0.5: # Horizontal
            rect1 = (x, y, w, h // 2)
            rect2 = (x, y + h // 2, w, h // 2)
            features.append((rect1, rect2))
        else: # Vertical
            rect1 = (x, y, w // 2, h)
            rect2 = (x + w // 2, y, w // 2, h)
            features.append((rect1, rect2))
    return features

def _calc_feature(self, integral_img, feature):
    """Calculates the value of a single Haar-like feature."""
    rect1, rect2 = feature
    
    def _sum_rect(r):
        x, y, w, h = r
        # A, B, C, D are corners of the rectangle
        # D----C
        # |    |
        # A----B
        A = integral_img[y, x]
        B = integral_img[y, x + w]
        C = integral_img[y + h, x + w]
        D = integral_img[y + h, x]
        return C + A - B - D

    return _sum_rect(rect1) - _sum_rect(rect2)

def _get_features(self, integral_img, bbox):
    """Extracts all feature values for a given bounding box."""
    x, y, w, h = bbox
    patch = cv2.resize(integral_img[y:y+h, x:x+w], self.roi_size)
    return np.array([self._calc_feature(patch, f) for f in self.features])

def _train(self, frame, bbox):
    """
    Implements discriminative classifier training using AdaBoost meta-learning algorithm.
    This procedure learns a weighted ensemble of weak Haar-feature classifiers that
    can distinguish the target object from background regions.
    
    Technical Algorithm:
    1. Feature Pool Generation: Create diverse set of Haar-like descriptors
    2. Sample Collection: Extract positive (target) and negative (background) training data
    3. AdaBoost Training: Iteratively select best weak classifiers and update sample weights
    4. Weight Updates: Exponential re-weighting emphasizes misclassified samples
    5. Ensemble Construction: Combine weak learners with confidence-weighted voting
    
    Mathematical Foundation:
    - Sample weights: D_{t+1}(i) = D_t(i) * exp(-α_t * y_i * h_t(x_i)) / Z_t
    - Classifier weight: α_t = 0.5 * log((1-ε_t)/ε_t) where ε_t is weighted error
    - Final hypothesis: H(x) = sign(Σ α_t * h_t(x))
    """
    x, y, w, h = bbox
    self.roi_size = (w, h)
    self.features = self._generate_features(w, h)
    integral_img = self._get_integral_image(frame)

    # 1. Get Positive Sample (the initial object)
    pos_features = self._get_features(integral_img, bbox)
    
    # 2. Generate Negative Training Set
    # Sample patches distant from target to create discriminative training data
    neg_samples = []
    while len(neg_samples) < self.n_samples:
        nx = np.random.randint(frame.shape[1] - w)
        ny = np.random.randint(frame.shape[0] - h)
        # Spatial separation constraint: enforce minimum distance to avoid label noise
        dist_sq = (nx - x)**2 + (ny - y)**2
        if dist_sq > self.search_radius**2:
            neg_samples.append(self._get_features(integral_img, (nx, ny, w, h)))

    # 3. AdaBoost Meta-Learning Algorithm
    # Combine feature vectors into training matrix with binary labels
    samples = np.vstack([pos_features] + neg_samples)
    labels = np.array([1] + [-1] * self.n_samples)
    
    # Initialize uniform sample weights for first boosting iteration
    weights = np.ones(len(labels)) / len(labels)
    
    # AdaBoost Iterative Weak Learner Selection
    self.classifiers = []
    for _ in range(self.n_features):
        min_error = float('inf')
        best_feature_idx = -1
        best_threshold = 0
        best_polarity = 1
        
        # Exhaustive search for optimal decision stump on current feature
        feature_vals = samples[:, _]
        sorted_indices = np.argsort(feature_vals)
        sorted_weights = weights[sorted_indices]
        sorted_labels = labels[sorted_indices]
        
        # Compute cumulative weighted label sums for efficient threshold search
        pos_seen = np.cumsum(sorted_weights * (sorted_labels == 1))
        neg_seen = np.cumsum(sorted_weights * (sorted_labels == -1))
        
        total_pos = pos_seen[-1]
        total_neg = neg_seen[-1]

        # Evaluate all possible threshold positions and polarities
        # Error for polarity = 1 (feature > threshold predicts positive)
        error1 = neg_seen + (total_pos - pos_seen)
        # Error for polarity = -1 (feature < threshold predicts positive)  
        error2 = pos_seen + (total_neg - neg_seen)
        
        # Select configuration minimizing weighted classification error
        all_errors = np.minimum(error1, error2)
        best_idx = np.argmin(all_errors)
        
        if all_errors[best_idx] < min_error:
            min_error = all_errors[best_idx]
            best_feature_idx = _
            best_threshold = feature_vals[sorted_indices[best_idx]]
            best_polarity = 1 if error1[best_idx] < error2[best_idx] else -1

        # Compute classifier confidence weight using AdaBoost formula
        # α_t = 0.5 * log((1 - ε_t) / ε_t) where ε_t is weighted error rate
        min_error = np.clip(min_error, 1e-10, 1 - 1e-10)  # Numerical stability
        alpha = 0.5 * np.log((1 - min_error) / min_error)
        
        # Handle numerical edge cases
        if np.isnan(alpha) or np.isinf(alpha):
            alpha = 0.0
        
        # Update sample weights using exponential loss gradient
        # D_{t+1}(i) = (D_t(i) / Z_t) * exp(-α_t * y_i * h_t(x_i))
        predictions = np.ones(len(samples))
        predictions[feature_vals * best_polarity < best_threshold * best_polarity] = -1
        weights *= np.exp(-alpha * labels * predictions)
        weights /= np.sum(weights)  # Normalize to maintain probability distribution
        
        self.classifiers.append({'feature_idx': best_feature_idx,
                                 'threshold': best_threshold,
                                 'polarity': best_polarity,
                                 'alpha': alpha})

def init(self, frame, bbox):
    """
    Initializes the tracker with the first frame and bounding box.
    
    Args:
        frame: The first frame of the video.
        bbox (tuple): The bounding box (x, y, w, h) of the object.
    """
    self.bbox = tuple(map(int, bbox))
    self._train(frame, self.bbox)

def update(self, frame):
    """
    Performs frame-to-frame tracking using discriminative classification and online learning.
    
    Technical Algorithm:
    1. Patch Sampling: Generate candidate locations in local neighborhood
    2. Feature Extraction: Compute Haar descriptors for each candidate patch  
    3. Classification: Apply boosted ensemble to score patch likelihood
    4. Target Localization: Select highest-confidence patch as new target location
    5. Online Learning: Update classifier weights using new positive/negative samples
    
    The online learning component adapts the appearance model to handle:
    - Gradual appearance changes (illumination, pose, deformation)
    - Partial occlusions and background clutter
    - Scale variations within fixed template size
    
    Args:
        frame: Input video frame for tracking
        
    Returns:
        tuple: Updated target bounding box (x, y, w, h) or None if tracking fails
    """
    if self.bbox is None:
        raise Exception("Tracker not initialized. Call init() first.")
    
    x, y, w, h = self.bbox
    integral_img = self._get_integral_image(frame)
    
    # 1. Exhaustive Local Search for Target Localization
    # Sample patches in spatial neighborhood using sliding window approach
    best_score = -float('inf')
    best_bbox = None
    
    # Generate candidate patch locations with fixed stride
    patches = []
    for dx in range(-self.search_radius, self.search_radius, 5):
        for dy in range(-self.search_radius, self.search_radius, 5):
            nx, ny = x + dx, y + dy
            # Enforce image boundary constraints
            nx = max(0, min(frame.shape[1] - w, nx))
            ny = max(0, min(frame.shape[0] - h, ny))
            patches.append((nx, ny, w, h))
    
    # 2. Discriminative Classification of Candidate Patches
    # Apply learned boosted classifier to score each patch
    confidences = []
    feature_vectors = []
    for patch_bbox in patches:
        features = self._get_features(integral_img, patch_bbox)
        feature_vectors.append(features)
        score = self.predict(features)  # Boosted ensemble prediction
        confidences.append(score)
        if score > best_score:
            best_score = score
            best_bbox = patch_bbox
    
    # Fallback to previous location if no confident detection found
    if best_bbox is None:
        best_bbox = self.bbox
    
    self.bbox = best_bbox
    
    # 3. Online Discriminative Learning with Temporal Regularization
    # Update classifier ensemble to adapt to appearance changes while preventing drift
    # This implements a form of online boosting with exponential forgetting
    
    # Generate positive training bag around new target location
    # Multiple samples provide robustness against localization noise
    pos_bag_patches = []
    px, py, pw, ph = self.bbox
    for dx in range(-5, 6, 2):  # Dense sampling in 5-pixel radius
        for dy in range(-5, 6, 2):
            nx, ny = px + dx, py + dy
            nx = max(0, min(frame.shape[1] - pw, nx))
            ny = max(0, min(frame.shape[0] - ph, ny))
            pos_bag_patches.append((nx, ny, pw, ph))
    
    # Extract features and select most confident positive sample
    pos_bag_features = [self._get_features(integral_img, p) for p in pos_bag_patches]
    pos_bag_scores = [self.predict(f) for f in pos_bag_features]
    
    # Maximum confidence sample serves as positive exemplar for online learning
    best_pos_sample = pos_bag_features[np.argmax(pos_bag_scores)]
    
    # Sample fresh negative examples from distant image regions
    # This maintains discriminative power against background drift
    neg_samples = []
    while len(neg_samples) < self.n_samples // 4:  # Reduced sample count for efficiency
        nx = np.random.randint(frame.shape[1] - w)
        ny = np.random.randint(frame.shape[0] - h)
        dist_sq = (nx - x)**2 + (ny - y)**2
        if dist_sq > self.search_radius**2 * 2:  # Increased separation for hard negatives
            neg_samples.append(self._get_features(integral_img, (nx, ny, w, h)))

    # Online Classifier Update using Temporal Exponential Smoothing
    # Each weak classifier is updated independently with new training data
    for i in range(len(self.classifiers)):
        clf = self.classifiers[i]
        
        # Evaluate current classifier on new training samples
        h_pos = 1 if best_pos_sample[clf['feature_idx']] * clf['polarity'] >= clf['threshold'] * clf['polarity'] else -1
        h_negs = np.array([1 if ns[clf['feature_idx']] * clf['polarity'] >= clf['threshold'] * clf['polarity'] else -1 for ns in neg_samples])
        
        # Compute classification accuracy on current samples
        correct_pos = 1 if h_pos == 1 else 0
        correct_negs = np.sum(h_negs == -1)
        
        # Update classifier weight based on current performance
        error = 1.0 - (correct_pos + correct_negs) / (1 + len(neg_samples))
        error = np.clip(error, 1e-10, 1 - 1e-10)  # Numerical stability
        new_alpha = 0.5 * np.log((1 - error) / error)
        
        if np.isnan(new_alpha) or np.isinf(new_alpha):
            new_alpha = 0.0
        
        # Exponential smoothing: α_new = (1-λ) * α_old + λ * α_computed
        # This provides temporal regularization preventing overfitting to recent frames
        clf['alpha'] = (1 - self.learning_rate) * clf['alpha'] + self.learning_rate * new_alpha

    return self.bbox

def predict(self, features):
    """
    Applies the learned boosted ensemble classifier to compute target confidence score.
    
    Technical Implementation:
    Uses weighted voting across all weak classifiers where each contributes its
    decision multiplied by its confidence weight (alpha). The final score represents
    log-likelihood ratio between target and background hypotheses.
    
    Mathematical Form:
    confidence = Σ(i=1 to T) α_i * h_i(x)
    where α_i is classifier weight and h_i(x) ∈ {-1, +1} is weak classifier decision
    
    Args:
        features: Haar feature vector for input patch
        
    Returns:
        float: Confidence score (higher values indicate higher target likelihood)
    """
    score = 0
    for clf in self.classifiers:
        val = features[clf['feature_idx']]
        h = 1 if val * clf['polarity'] >= clf['threshold'] * clf['polarity'] else -1
        # Check for invalid alpha values
        if not np.isnan(clf['alpha']) and not np.isinf(clf['alpha']):
            score += clf['alpha'] * h
    return score

# Bind methods to ObjectTracker class
ObjectTracker._get_integral_image = _get_integral_image
ObjectTracker._generate_features = _generate_features
ObjectTracker._calc_feature = _calc_feature
ObjectTracker._get_features = _get_features
ObjectTracker._train = _train
ObjectTracker.init = init
ObjectTracker.update = update
ObjectTracker.predict = predict


def show_camera_with_tracker(source=0, track_everything=False):
  
    cap = cv2.VideoCapture(source)
    tracker = YOLOTracker(track_everything=track_everything) if YOLO_AVAILABLE else ObjectTracker()
    
    ret, first_frame = cap.read()
    if not ret:
        print("Failed to read from video source")
        return
    
    # Initialize tracker
    if isinstance(tracker, YOLOTracker):
        success = tracker.init(first_frame)
        if not success:
            cap.release()
            cv2.destroyAllWindows()
            return
    else:
        # Fallback to manual selection for basic tracker
        cv2.putText(first_frame, "Select object to track, then press SPACE to continue", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow('Object Tracker', first_frame)
        
        print("Select the object you want to track in the video window, then press SPACE to start tracking")
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC to exit
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord(' '): 
                initBB = cv2.selectROI('Object Tracker', first_frame, fromCenter=False, showCrosshair=True)
                if initBB[2] > 0 and initBB[3] > 0:
                    tracker.init(first_frame, initBB)
                    break
                else:
                    print("Invalid selection. Try again by pressing SPACE")

    frame_count = 0
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), 
              (0, 255, 255), (128, 0, 128), (255, 165, 0), (0, 128, 0), (128, 128, 0)]
    
    # Initialize FPS tracking
    start_time = time.time()
    fps_history = []
    fps_window_size = 30  # Calculate average over last 30 frames
    
    while True:
        frame_start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        result = tracker.update(frame)
        
        # Calculate FPS
        frame_end_time = time.time()
        frame_time = frame_end_time - frame_start_time
        current_fps = 1.0 / frame_time if frame_time > 0 else 0
        
        # Maintain FPS history for rolling average
        fps_history.append(current_fps)
        if len(fps_history) > fps_window_size:
            fps_history.pop(0)
        
        # Calculate average FPS using numpy
        avg_fps = np.mean(fps_history)
        overall_fps = frame_count / (frame_end_time - start_time) if frame_count > 0 else 0
        
        if isinstance(tracker, YOLOTracker) and tracker.track_everything:
            # Multi-object tracking display
            if result:
                for i, (tracker_id, track_info) in enumerate(result.items()):
                    bbox = track_info['bbox']
                    class_name = track_info['class_name']
                    
                    if bbox is not None:
                        (x, y, w, h) = [int(v) for v in bbox]
                        color = colors[i % len(colors)]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(frame, f"ID{tracker_id}: {class_name}", (x, y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Show tracking info
            cv2.putText(frame, f"Tracking {len(tracker.trackers)} objects", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            # Single object tracking display
            if result is not None:
                if isinstance(result, dict):  # Multi-object result but single mode
                    for track_info in result.values():
                        bbox = track_info['bbox']
                        if bbox is not None:
                            (x, y, w, h) = [int(v) for v in bbox]
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            break
                else:  # Single bbox result
                    (x, y, w, h) = [int(v) for v in result]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Show detection info if using YOLO
                    if isinstance(tracker, YOLOTracker) and hasattr(tracker, 'target_class') and tracker.target_class is not None:
                        class_name = tracker.yolo_model.names[tracker.target_class] if tracker.yolo_model else "Unknown"
                        cv2.putText(frame, f"Tracking: {class_name}", (x, y - 25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Tracking failure", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Show frame info
        if isinstance(tracker, YOLOTracker):
            detection_status = "YOLO" if frame_count % tracker.detection_interval == 0 else "Tracking"
            mode_text = "Multi-Track" if tracker.track_everything else "Single-Track"
            cv2.putText(frame, f"Mode: {mode_text} | {detection_status} | Frame: {frame_count}", 
                       (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display FPS information
        cv2.putText(frame, f"Avg FPS: {avg_fps:.1f} | Overall: {overall_fps:.1f}", 
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Display control instructions
        if isinstance(tracker, YOLOTracker) and tracker.track_everything:
            cv2.putText(frame, "ESC/Q: Exit | C: Clear all | R: Reset", 
                       (frame.shape[1] - 300, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "ESC/Q: Exit | R: Reset", 
                       (frame.shape[1] - 150, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imshow('Object Tracker', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC to exit
            break
        elif key == ord('q'):  # Q to quit
            break
        elif key == ord('c') and isinstance(tracker, YOLOTracker) and tracker.track_everything:
            # C to clear all trackers in multi-object mode
            tracker.trackers.clear()
            print("Cleared all trackers")
        elif key == ord('r') and isinstance(tracker, YOLOTracker):
            # R to reset/reinitialize tracker
            print("Reinitializing tracker...")
            success = tracker.init(frame)
            if not success:
                print("Failed to reinitialize tracker")

    cap.release()
    cv2.destroyAllWindows()

# --- Main execution block to demonstrate the tracker ---
if __name__ == '__main__':
    print("Advanced Object Tracker with YOLOv11 Integration")
    print("================================================")
    print("\nChoose input source:")
    print("1. Webcam (real-time)")
    print("2. Video file")
    choice = input("Enter 1 or 2: ").strip()
    track_everything = False

    if choice == "1":
        show_camera_with_tracker(0, track_everything)
    elif choice == "2":
        video_path = input("Enter path to video file: ").strip()
        show_camera_with_tracker(video_path, track_everything)
    else:
        print("Invalid choice.")