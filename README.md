# Intel Image Classifier

A complete project that classifies natural scene images into 6 categories
using custom CNN models built with PyTorch and TensorFlow/Keras, served via a Flask web application.


## Project Structure

```
project/
├── train.py               Main training script (PyTorch + TensorFlow)
├── app.py                 Flask web server
├── requirements.txt       Python dependencies
├── README.md              This file
│
├── models/                Saved model files (created after training)
│   ├── Tikome_Nana_model.pth     (PyTorch model)
│   └── Tikome_Nana_model.keras   (TensorFlow model)
│
├── templates/
│   └── index.html         Frontend HTML page
│
└── static/
    ├── css/
    │   └── style.css      Page styling
    └── js/
        └── app.js         Frontend JavaScript logic
```


# 1. Install dependencies

pip install -r requirements.txt


# 2. Train the models


# Train PyTorch model
python train.py --model pytorch --epochs 15

# Train TensorFlow model
python train.py --model tensorflow --epochs 15

# Custom options
python train.py --model pytorch --epochs 20 --batch_size 64 --data_dir /path/to/intel_dataset


# 3. Run the web app

python app.py

Then open: **http://localhost:5000**

---

#Dataset: Intel Image Classification

| Split     | Location                             | Images  |
|-----------|--------------------------------------|---------|
| Train     | `intel_dataset/seg_train/seg_train/` | ~14,000 |
| Test      | `intel_dataset/seg_test/seg_test/`   | ~3,000  |


**6 Classes** : buildings, forest, glacier, mountain, sea, street




#Deployment

Using Render.com



#Expected Results

After 15 epochs on the full Intel dataset:
- PyTorch CNN: ~ 85-90% test accuracy
- TensorFlow/Keras CNN: ~85-90% test accuracy

Training time (CPU): ~ 45-90 minutes
Training time (GPU): ~ 5-10 minutes
