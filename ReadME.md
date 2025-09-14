Quantum-Aided Acne Severity Detection: A Framework with Enhanced Sketch-Based Imaging

We present an GAN-powered image processing framework for enhancing acne severity. Our approach leverages Sketch Transformation, Super Resolution and Quantum Networks to effectively enhance sketches and classify acne severity classes.

Process Flow
Image

Installation
To install the required libraries, you can use the following command:
pip install -r requirements.txt

Module Breakdown
SketchModule
Converts input photos into clean line sketches using a Pix2Pix GAN.

SuperResolutionModule
Enhances and deblurs sketches with SRGAN-based super-resolution, producing sharper high-resolution sketches.

QuantumModule
Classifies the enhanced sketches with a 10-qubit hybrid quantum-classical model (7 amplitude + 3 angle encoding), enabling accurate acne severity prediction.

Technologies Used
SketchModule (Photo → Sketch)
    Frameworks: PyTorch, torchvision
    Model: Pix2Pix GAN
    Purpose: Translates photos into clean sketches

SuperResolutionModule (Sketch → HR Sketch)
    Frameworks: TensorFlow, TensorLayerX
    Model: SRGAN (Super-Resolution GAN)
    Purpose: Enhances and sharpens sketches using contour-aware upscaling

QuantumModule (Sketch → Classification)
    Frameworks: Qiskit, Qiskit Machine Learning, PyTorch
    Model: Hybrid Variational Quantum Classifier (10-qubits: 7 amplitude + 3 angle encoding)
    Purpose: Classifies enhanced sketches into acne severity levels

Usage
Clone the repository using

Verify the folder structure
Root/
│
├── QuantumModule/                
│   ├── data\superesolvedsketches
│   ├── helpers
│   │   ├── feature_extractor.py
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

Datasets Used
    Acne recognition dataset: https://drive.google.com/drive/folders/1SA3nboBWxZHm04rsq--ezd7ZwnFxfLc6?usp=sharing
    CUHK: https://drive.google.com/drive/folders/1SB4W-9FO-IAy91z2cRxZHLWiUD1fjJgY?usp=sharing
    AR: https://drive.google.com/drive/folders/1t3EgNgA1PdJZqDcenVCpjmXvVQ8VwouI?usp=sharing
    XM2GTS: https://drive.google.com/drive/folders/1l_wPPVDg-_7GM0QIqh-BXx10QAWCivt7?usp=sharing

How to Run
For Running SketchModule:
cd sketchmodule
Organize your dataset as follows: (Example folder structure):
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

python image_to_sketch.py
The results are present in the /code/results folder.

For Running SuperResolutionModule:
cd SuperResolutionModule
Organize your dataset as follows: (Example folder structure):
data/
              HR/
                   01.jpg
                   02.jpg
                   ...
              LR/
                   01.jpg
                   02.jpg
                    ...
python superresolute_runner.py

For Running SuperResolutionModule:
cd QuantumModule
Add your superesolvedsketches in the data/superesolvedsketches folder (Also add your acne labels as required).
cd helpers
python feature_extractor.py


