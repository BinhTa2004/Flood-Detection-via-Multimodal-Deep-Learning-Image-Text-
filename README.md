# 🌊 Flood Detection via Multimodal Deep Learning
Developed a multimodal flood classification system using **Vision Transformer (ViT)** for image feature extraction and **BERT** for encoding contextual information from text (titles, descriptions, tags). Designed a fusion model to combine both modalities and trained a classifier to detect flood-related scenarios. Built an end-to-end pipeline with preprocessing, fine-tuning, evaluation, and threshold optimization to enhance performance.

---

## 🛠️ Technologies
- Python  
- PyTorch  
- BERT (HuggingFace Transformers)  
- Vision Transformer (ViT)  
- Scikit-learn  
- Computer Vision  
- Natural Language Processing (NLP)  

---

## ✨ Project Highlights
- Multimodal input: combines visual (images) and textual (title, description, tags) data.  
- Feature extraction using pretrained ViT and BERT models.  
- Fusion model concatenates both modalities into a unified representation.  
- Classification head trained to detect flood-related scenarios.  
- Evaluation using Mean Average Precision (MAP).  
- Achieved strong performance: MAP@250 = 93.0 (public), 91.0 (private).  
- Includes threshold optimization and stratified K-Fold cross-validation.  

---

## ⚙️ Setup
Clone the repository:
```bash
git clone https://github.com/BinhTa2004/Flood-Detection.git
cd Flood-Detection
```

---

## 📦 Data
This project requires both image and text data:
- **Images**: flood-related photos used for visual analysis.
- **Text metadata**: accompanying information such as:
  - `title`
  - `description`
  - `user_tags`
  - `image_id` (used to match with image files)
  - `label` (binary: 1 = flood, 0 = non-flood)

Due to size or licensing restrictions, the dataset is not included.  
Please refer to the competition or data source to obtain the full dataset.

Link: https://www.kaggle.com/datasets/binhhhhhhhhh/dpl-2025
