import os
import cv2
import csv
import numpy as np
from numpy.linalg import norm
import datetime as datetime

import torch
import torchvision.transforms as T
from facenet_pytorch import InceptionResnetV1, MTCNN

def get_embedding(image_rgb):
    """
    Extract embedding from an image
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    embedder = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    detector = MTCNN(
    keep_all=True,  
    min_face_size=20,
    thresholds=[0.6, 0.7, 0.7],
    device=device 
    )

    transform = T.Compose([
    T.ToPILImage(),
    T.Resize((160, 160)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    try:
        boxes, _ = detector.detect(image_rgb)
        
        if boxes is not None and len(boxes) > 0:
            x1, y1, x2, y2 = boxes[0].astype(int)
            
            if x2 <= x1 or y2 <= y1:
                return None, None
            
            h, w = image_rgb.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            face = image_rgb[y1:y2, x1:x2]
            
            if face.shape[0] < 20 or face.shape[1] < 20:
                return None, None
            
            face_tensor = transform(face).unsqueeze(0).to(device)
            
            with torch.no_grad():
                embedding = embedder(face_tensor)
            
            return embedding.cpu().numpy().flatten(), (x1, y1, x2-x1, y2-y1)
    
    except Exception as e:
        print(f"extraction Error: {e}")
    

def build_database(dataset_path="dataset"):
    """
    Create embedding's database
    """
    embeddings_db = {}
    
    if not os.path.exists(dataset_path):
        print(f" Directory not found: {dataset_path}")
        return embeddings_db
    
    persons = [p for p in os.listdir(dataset_path) 
               if os.path.isdir(os.path.join(dataset_path, p))]
    
    if not persons:
        print(" Directory is empty")
        return embeddings_db
    
    print(f"Treatement of {len(persons)} persons...")
    
    for person in persons:
        person_dir = os.path.join(dataset_path, person)
        embeds = []
        
        image_files = [f for f in os.listdir(person_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print(f"  {person}: Pas d'images")
            continue
        
        print(f"  {person}: {len(image_files)} images")
        
        for img_file in image_files[:30]:  
            img_path = os.path.join(person_dir, img_file)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            embedding, _ = get_embedding(img_rgb)
            
            if embedding is not None:
                embeds.append(embedding)
        
        # Mean embedding
        if embeds:
            avg_embed = np.mean(embeds, axis=0)
            embeddings_db[person] = avg_embed
            print(f"{len(embeds)} valid embeddings")
        else:
            print(f"NO valid embedding")
    
    # Saving
    if embeddings_db:
        np.save("embeddings.npy", embeddings_db)
        print(f"\n Created base : {len(embeddings_db)} persons")
    else:
        print("\n empty base!")
    
    return embeddings_db


def real_time_attendance(
    embeddings_path="test/embeddings.npy",
    csv_file="attendance.csv",
    threshold=0.6):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    mtcnn = MTCNN(keep_all=True, device=device)
    facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    data = np.load(embeddings_path, allow_pickle=True).item()
    known_names = list(data.keys())
    known_embeddings = np.array(list(data.values()))

    known_embeddings = known_embeddings / norm(known_embeddings, axis=1, keepdims=True)

    attendance = {name: "Absent" for name in known_names}

    cap = cv2.VideoCapture(0)
    print("Press 'q' to stop")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(rgb)

        if boxes is not None:
            faces = mtcnn.extract(rgb, boxes, save_path=None)

            for box, face in zip(boxes, faces):
                face = face.unsqueeze(0).to(device)

                with torch.no_grad():
                    embedding = facenet(face).cpu().numpy()[0]

                embedding = embedding / norm(embedding)

                scores = np.dot(known_embeddings, embedding)

                scores_per_person = {
                    known_names[i]: float(scores[i])
                    for i in range(len(known_names))
                }

                best_idx = np.argmax(scores)
                best_score = scores[best_idx]

                if best_score >= threshold:
                    name = known_names[best_idx]
                    attendance[name] = "Present"
                    color = (0, 255, 0)
                    label = f"{name} ({best_score:.2f})"
                else:
                    name = "Unknown"
                    color = (0, 0, 255)
                    label = f"Unknown ({best_score:.2f})"

                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

        cv2.imshow("FaceNet Attendance", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Status", "Time"])
        for name, status in attendance.items():
            writer.writerow([name, status, time_now])

    print("Attendance saved to", csv_file)

