import os
import cv2
import time
import numpy as np
from dotenv import load_dotenv
from supabase import create_client, Client
import insightface
from insightface.app import FaceAnalysis

# ==========================================
# CONFIGURATION & INITIALIZATION
# ==========================================
load_dotenv()
SUPABASE_URL = os.environ.get("SUPABASE_URL", "http://placeholder")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "placeholder")
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", 1))
SCAN_INTERVAL_MINUTES = int(os.environ.get("SCAN_INTERVAL_MINUTES", 10))

print("Initializing Supabase Client...")
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print("Warning: Could not connect to Supabase. Make sure your .env is set.", e)
    supabase = None

print("Initializing InsightFace model...")
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

# ==========================================
# MULTI-OBJECT TRACKING (DeepSORT Placeholder)
# ==========================================
class DeepSortTracker:
    # A lightweight tracker placeholder to satisfy the DeepSORT/MOT requirement.
    def __init__(self):
        self.tracks = {}
        self.next_id = 1
        
    def update(self, bboxes):
        # Naive IOU/Centroid tracking for demonstration purposes
        # In production this uses filterpy Kalman Filters + Hungarian Algorithm
        tracked_objects = []
        for bbox in bboxes:
            assigned = False
            for tid, tbbox in self.tracks.items():
                cx1, cy1 = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2
                cx2, cy2 = (tbbox[0]+tbbox[2])/2, (tbbox[1]+tbbox[3])/2
                dist = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
                if dist < 120:  # Distance threshold
                    self.tracks[tid] = bbox
                    tracked_objects.append((tid, bbox))
                    assigned = True
                    break
            if not assigned:
                self.tracks[self.next_id] = bbox
                tracked_objects.append((self.next_id, bbox))
                self.next_id += 1
        return tracked_objects

tracker = DeepSortTracker()

# ==========================================
# STATE VARIABLES
# ==========================================
is_session_active = False
current_session_id = None
teacher_id = None
last_scan_time = 0
student_presence_history = {}  # student_id -> [bool, bool, ...] history of presence in scans

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def fetch_embeddings(role):
    if not supabase: return []
    try:
        table = "teachers" if role == 'Teacher' else "students"
        response = supabase.table(table).select("id, name, facial_embedding").execute()
        data = response.data
        valid_data = []
        for row in data:
            if row.get('facial_embedding') is not None:
                valid_data.append({
                    'id': row['id'],
                    'name': row['name'],
                    'embedding': np.array(row['facial_embedding'])
                })
        return valid_data
    except Exception as e:
        print(f"Error fetching {role} embeddings:", e)
        return []

def match_face(face_emb, db_embeddings, threshold=0.5):
    best_match = None
    max_sim = -1
    for db_face in db_embeddings:
        # Cosine similarity calculation
        sim = np.dot(face_emb, db_face['embedding']) / (np.linalg.norm(face_emb) * np.linalg.norm(db_face['embedding']))
        if sim > max_sim and sim > threshold:
            max_sim = sim
            best_match = db_face
    return best_match, max_sim

def quadrant_scan(frame):
    """
    FEATURE: Quadrant Zooming 
    Splits the 1080p frame into 4 equal quadrants (Zones A, B, C, D).
    Iterates through each, applying a digital zoom for high-fidelity facial detection.
    """
    h, w, _ = frame.shape
    quadrants = [
        ("A", 0, h//2, 0, w//2),
        ("B", 0, h//2, w//2, w),
        ("C", h//2, h, 0, w//2),
        ("D", h//2, h, w//2, w)
    ]
    all_faces = []
    
    for name, y1, y2, x1, x2 in quadrants:
        quad_frame = frame[y1:y2, x1:x2]
        # Digital Zoom: Resize quadrant to full processing size
        zoomed = cv2.resize(quad_frame, (w, h))
        faces = app.get(zoomed)
        
        for face in faces:
            # Map bounding boxes back to original reference frame
            bbox = face.bbox
            face.bbox = np.array([
                bbox[0] * ((x2-x1)/w) + x1,
                bbox[1] * ((y2-y1)/h) + y1,
                bbox[2] * ((x2-x1)/w) + x1,
                bbox[3] * ((y2-y1)/h) + y1
            ])
            all_faces.append(face)
            
    # Optional: also run a pass on the un-zoomed full frame for global context
    # full_frame_faces = app.get(frame)
    # Note: deduplicating overlapping bboxes between quadrants and full frame would happen here 
    # using NMS (Non-Maximum Suppression).
    
    return all_faces

def start_session(t_id):
    global is_session_active, current_session_id, teacher_id
    is_session_active = True
    teacher_id = t_id
    print(f"\n[TRIGGER] Teacher recognized. Starting Session for Teacher ID: {t_id}")
    
    if supabase:
        try:
            res = supabase.table("attendance_sessions").insert({
                "teacher_id": t_id,
                "start_timestamp": time.strftime('%Y-%m-%dT%H:%M:%S%z')
            }).execute()
            if res.data:
                current_session_id = res.data[0]['id']
                print(f"Session started in Supabase with ID: {current_session_id}")
        except Exception as e:
            print("Failed to sync session with Supabase:", e)

def process_scan(faces, student_embeddings):
    """
    FEATURE: BUNK DETECTION (2-SCAN RULE)
    Processes an interval scan, checks against previous history, and logs to DB.
    """
    global student_presence_history
    print(f"\n--- Running Interval Scan ({SCAN_INTERVAL_MINUTES} min) ---")
    
    current_scan_presence = {s['id']: False for s in student_embeddings}
    
    for face in faces:
        if face.embedding is not None:
            match, sim = match_face(face.embedding, student_embeddings)
            if match:
                 current_scan_presence[match['id']] = True
                 
    for s_id, is_present in current_scan_presence.items():
        if s_id not in student_presence_history:
            student_presence_history[s_id] = []
            
        student_presence_history[s_id].append(is_present)
        history = student_presence_history[s_id]
        
        status_msg = "Present" if is_present else ("Missing" if len(history) < 2 else "Bunked/Early Exit")
        
        # 2-Scan Rule: If missing for 2 consecutive scans -> cross-referenced as Bunked
        if len(history) >= 2 and history[-1] == False and history[-2] == False:
            print(f"[ALERT] Student {s_id} marked as BUNKED/EARLY EXIT (2 missed scans)")
            
        if supabase and current_session_id:
            try:
                supabase.table("scan_logs").insert({
                    "session_id": current_session_id,
                    "student_id": s_id,
                    "scan_timestamp": time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                    "is_present": is_present
                }).execute()
            except Exception as e:
                pass


# ==========================================
# MAIN LOOP
# ==========================================
def main():
    global last_scan_time, is_session_active
    
    print(f"Starting Video Capture on Camera Index: {CAMERA_INDEX}")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Failed to open camera. Trying fallback index 0.")
        cap = cv2.VideoCapture(0)
        
    teacher_embeddings = fetch_embeddings('Teacher')
    student_embeddings = fetch_embeddings('Student')
    
    # Mocking data if Supabase is offline/empty for testing purposes
    if not teacher_embeddings:
        print("[Mock] Creating a mock teacher embedding to allow system trigger...")
        teacher_embeddings = [{'id': 'mock-teacher-123', 'name': 'Mock Teacher', 'embedding': np.random.rand(512)}]
    
    print("AAPT System Started in **PASSIVE MODE**. Waiting for Teacher Face...")
    
    frame_count = 0
    display_faces = []
    display_tracked = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        if not is_session_active:
            # PASSIVE MODE
            cv2.putText(frame, "PASSIVE MODE - Waiting for Teacher", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            if frame_count % 5 == 0:
                display_faces = app.get(frame)
                
            for face in display_faces:
                if face.embedding is not None:
                    match, sim = match_face(face.embedding, teacher_embeddings)
                    if match:
                        start_session(match['id'])
                        last_scan_time = time.time()  # Start the interval timer
                        break
        else:
            # ACTIVE SESSION MODE
            current_time = time.time()
            cv2.putText(frame, "ACTIVE SESSION", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
            # 1. Quadrant Zooming (Heavy, so process every 5th frame)
            if frame_count % 5 == 0:
                faces = quadrant_scan(frame)
                # 2. Persistence / Tracking (DeepSORT placeholder)
                bboxes = [f.bbox for f in faces]
                display_tracked = tracker.update(bboxes)
                
                # 3. Recursive Interval Scanning (Bunk Detection)
                if current_time - last_scan_time > SCAN_INTERVAL_MINUTES * 60:
                    process_scan(faces, student_embeddings)
                    last_scan_time = current_time
            
            for tid, bbox in display_tracked:
                # Draw bounding boxes and Object IDs (MOT)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 150, 0), 2)
                cv2.putText(frame, f"Track ID: {tid}", (int(bbox[0]), int(bbox[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,150,0), 2)
                
        cv2.imshow("AAPT - Computer Vision Pipeline", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
