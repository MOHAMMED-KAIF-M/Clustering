import os
import cv2
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

def load_images(input_dir):
    image_paths = [os.path.join(input_dir, f) for f in sorted(os.listdir(input_dir)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return image_paths

def main():
    input_dir = "real_input"
    image_paths = load_images(input_dir)
    n = len(image_paths)

    orb = cv2.ORB_create(nfeatures=2500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()

    orb_sim = np.zeros((n, n), dtype=np.float32)
    clip_sim = np.zeros((n, n), dtype=np.float32)

    embeddings = []
    orb_desc = []
    
    for path in image_paths:
        img = cv2.imread(path)
        h, w = img.shape[:2]
        if max(h, w) > 800:
            scale = 800 / float(max(h, w))
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        kp, des = orb.detectAndCompute(gray, None)
        orb_desc.append(des)
        
        fake_rgb = cv2.merge([gray, gray, gray])
        pil_img = Image.fromarray(fake_rgb)
        inputs = clip_processor(images=pil_img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = clip_model.vision_model(**inputs["pixel_values"].unsqueeze(0) if inputs["pixel_values"].dim() == 3 else inputs)
            feat = outputs.pooler_output
        feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
        embeddings.append(feat.cpu().numpy()[0])

    for i in range(n):
        for j in range(i, n):
            if i == j:
                continue
            # ORB
            des1, des2 = orb_desc[i], orb_desc[j]
            o_score = 0.0
            if des1 is not None and des2 is not None and len(des1) >= 2 and len(des2) >= 2:
                m12 = bf.knnMatch(des1, des2, k=2)
                good12 = sum(1 for p in m12 if len(p) == 2 and p[0].distance < 0.80 * p[1].distance)
                m21 = bf.knnMatch(des2, des1, k=2)
                good21 = sum(1 for p in m21 if len(p) == 2 and p[0].distance < 0.80 * p[1].distance)
                o_score = max(good12, good21) / float(min(len(des1), len(des2)))
            
            orb_sim[i, j] = o_score
            orb_sim[j, i] = o_score
            
            # CLIP
            c_score = float(np.dot(embeddings[i], embeddings[j]))
            clip_sim[i, j] = c_score
            clip_sim[j, i] = c_score

    # Absolute Normalization
    # ORB typically saturates around 0.10 for an amazing geometric match
    orb_norm = np.clip(orb_sim / 0.10, 0.0, 1.0)
    
    # CLIP typically ranges 0.70 to 0.95
    clip_norm = np.clip((clip_sim - 0.70) / (0.95 - 0.70), 0.0, 1.0)
    
    combined = 0.6 * orb_norm + 0.4 * clip_norm
    np.fill_diagonal(combined, 1.0)

    print("\n=== Combined Similarity (Absolute Scaling) ===\n")
    for i in range(n):
        row = f"[{i:>2}] "
        for j in range(n):
            row += f"{combined[i][j]:.3f} "
        print(row)

if __name__ == "__main__":
    main()
