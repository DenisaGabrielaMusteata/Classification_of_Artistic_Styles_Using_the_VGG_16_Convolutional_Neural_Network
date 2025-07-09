# Artistic Style Classification Using VGG-16

This is a **Machine Learning project developed as part of a university course** at the Gheorghe Asachi Technical University of Iași, Faculty of Automatic Control and Computer Engineering. The goal is to classify images of paintings based on their artistic styles using a deep learning model—**VGG-16**, with transfer learning applied to the **WikiArt dataset**.

## Author

- **Denisa-Gabriela Musteață**
- Department of Automatic Control and Applied Informatics
- Email: denisa-gabriela.musteata@student.tuiasi.ro

---

## Project Overview

The project uses a **pre-trained VGG-16 Convolutional Neural Network**, fine-tuned on a custom dataset extracted from WikiArt, to classify artwork into six artistic styles:
- Baroque
- Realism
- Impressionism
- Abstract Expressionism
- Cubism
- Ukiyo-e

## Dataset

- **Source:** [WikiArt Dataset](https://www.wikiart.org/)
- **Classes Used:** A reduced subset of 6 major styles (~2,000 images)
- **Preprocessing:** Image resizing (224×224), normalization, dataset splitting (70% train / 15% validation / 15% test)

---

## Methodology

### Model
- **Architecture:** VGG-16
- **Transfer Learning:** Used pretrained weights from ImageNet
- **Custom Layers:** New fully connected layers for classification
- **Frozen Layers:** Convolutional base frozen to speed up training

### Two-Level Classification
To address confusion among visually similar styles (Baroque, Realism, Impressionism), a **two-level classification approach** was used:
1. Train model with these styles merged into one group
2. Train a second model to distinguish between them

---

## Experimental Results

| Experiment                        | Train Accuracy | Test Accuracy | Validation Accuracy |
|----------------------------------|----------------|----------------|----------------------|
| All 6 classes                    | 90.04%         | 59.56%         | 60.13%               |
| 5 classes (no Baroque)          | 95.37%         | 78.57%         | 78.23%               |
| 4 classes (no Baroque, Realism) | 98.37%         | 91.70%         | 86.27%               |
| Two-Level (combined + split)    | ~95%           | up to 96.97%   | ~82.70%              |

---

## Technologies Used

- MATLAB (Deep Learning Toolbox)
- VGG-16 (Pretrained CNN)
- Image Processing
- Transfer Learning
- Confusion Matrix Evaluation

---

## Future Work

- Add more balanced and diverse samples per style
- Explore data augmentation and advanced image filters
- Replace VGG-16 with more efficient models like ResNet or EfficientNet
- Deploy as a web-based art classification app

---

## Academic Context

This repository was developed as part of the coursework in **Machine Learning** at the **Technical University "Gheorghe Asachi" of Iași**. The project combines computer vision and art, offering insight into how **deep learning can support cultural and creative domains**.

---

> *For educational and non-commercial use only.*
