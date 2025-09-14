# Quantum-Aided Acne Severity Detection: A Framework with Enhanced Sketch-Based Imaging
#### Philomina Princiya Mascarenhas, Sannidhan M S, Jason Elroy Martis


We present a **Quantum powered image processing framework** for enhancing acne severity detection.  
Our approach leverages **Sketch Transformation, Super Resolution, and Quantum Networks** to effectively enhance sketches and classify acne severity classes.

## 📌 Process Flow



## ⚙️ Installation
Install the required libraries with:

```bash
pip install -r requirements.txt
```

## Module Breakdown
1. SketchModule: Converts input photos into clean line sketches using a Pix2Pix GAN.
2. SuperResolutionModule: Enhances and deblurs sketches with SRGAN-based super-resolution, producing sharper high-resolution sketches.
3. QuantumModule: Classifies the enhanced sketches with a 10-qubit hybrid quantum-classical model (7 amplitude + 3 angle encoding), enabling accurate acne severity prediction.

## Technologies Used
1. SketchModule (Photo → Sketch)
    Frameworks: PyTorch, torchvision
    Model: Pix2Pix GAN
    Purpose: Translates photos into clean sketches

2. SuperResolutionModule (Sketch → HR Sketch)
    Frameworks: TensorFlow, TensorLayerX
    Model: SRGAN (Super-Resolution GAN)
    Purpose: Enhances and sharpens sketches using contour-aware upscaling

3. QuantumModule (Sketch → Classification)
    Frameworks: Qiskit, Qiskit Machine Learning, PyTorch
    Model: Hybrid Variational Quantum Classifier (10-qubits: 7 amplitude + 3 angle encoding)
    Purpose: Classifies enhanced sketches into acne severity levels

## Folder Structure
```
Root/
│
├── QuantumModule/                
│   ├── data/superesolvedsketches/
│   ├── helpers/
│   │   └── feature_extractor.py
│   └── quantum_module.py                              
│
├── SketchModule/ 
│   ├── sketch_to_image.py                
│   ├── code/
│   │   ├── combine_A_and_B.py    
│   │   └── train.py              
│   └── data/dataset/
│       ├── photos/    
│       ├── face2sketch/             
│       └── sketches/             
│
├── SuperResolutionModule/        
│   ├── data/
│   │   ├── HR/                   
│   │   └── LR/                   
│   ├── srgan.py                  
│   ├── train.py     
│   ├── config.py  
│   ├── vgg.py                
│   └── superresolute_runner.py   
│
└── requirements.txt
```

## Datasets Used
1. Acne recognition dataset: https://drive.google.com/drive/folders/1SA3nboBWxZHm04rsq--ezd7ZwnFxfLc6?usp=sharing
2. CUHK: https://drive.google.com/drive/folders/1SB4W-9FO-IAy91z2cRxZHLWiUD1fjJgY?usp=sharing
3. AR: https://drive.google.com/drive/folders/1t3EgNgA1PdJZqDcenVCpjmXvVQ8VwouI?usp=sharing
4. XM2GTS: https://drive.google.com/drive/folders/1l_wPPVDg-_7GM0QIqh-BXx10QAWCivt7?usp=sharing

## How to Run
### 1. SketchModule
```
cd SketchModule
```
Organize your dataset as:
```
data/
   dataset/
      photos/
         photo_01.jpg
         photo_02.jpg
         ...
      sketches/
         sketch_01.jpg
         sketch_02.jpg
         ...
```
Execute: ```python image_to_sketch.py```
Results will appear in: ```SketchModule/code/results/```

### 2. SuperResolutionModule
```
cd SuperResolutionModule
```
Organize your dataset as:
```
data/
   HR/
      01.jpg
      02.jpg
      ...
   LR/
      01.jpg
      02.jpg
      ...
```
Execute: ```python superresolute_runner.py```
Results will appear in: ```data/HR```

### 3. QuantumModule
```
cd QuantumModule
```
Prepare your data:
1. Place super-resolved sketches in: data/superesolvedsketches/
2. Add acne labels in: labels.csv

Extract features: 
```
cd helpers
python feature_extractor.py
```
Train and classify: ```python quantum_module.py```