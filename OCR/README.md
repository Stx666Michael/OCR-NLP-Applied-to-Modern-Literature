# MSc Individual Project
**Title**: Chinese Character Recognition: Applied to Modern Literature

**Author**: Michael Song (ms423)

**Supervisor**: Sibo Cheng

**Second Marker**: Pancham Shukla

## Introduction
This project focuses on developing an Optical Character Recognition (OCR) system for digitizing Funü Zazhi, a historical Chinese magazine published between 1915 and 1931, with the aim of enabling gender discourse analysis using Natural Language Processing (NLP). The Funü Zazhi database comprises over 36,000 scanned pages, characterized by low image resolution, diverse text layouts, and varying image quality, which pose significant challenges for conventional OCR tools. To address them, we developed an OCR system with text detection, recognition, and ordering modules, tailored to the specific needs of the dataset.

The system was trained using synthetically generated data that simulates the characteristics of Funü Zazhi, due to the lack of relevant annotated dataset. Through a series of experiments and optimization, the OCR system achieved an average inference speed of 2.88s/image, a mAP@[0.5:0.95] of 0.7, a Character Error Rate (CER) of 0.31, and a BLEU-4 score of 0.56 on test images of Funü Zazhi, outperforming 4 state-of-the-art OCR tools by a large margin. Further analysis using a confidence score estimated that around 30\% of the images in Funü Zazhi can achieve high accuracy, with CER around 0.15 and BLEU-4 score around 0.73, providing a solid foundation for subsequent NLP analysis.

Future work includes improving real data collection, refining text ordering algorithms, enhancing the synthetic data generation process, and expanding the system to support digitization of other historical publications. The project contributes to the preservation and analysis of cultural heritage, demonstrating the potential of OCR in facilitating interdisciplinary research between humanities and computer science.

## Project Structure
The project is structured as follows, where all the references are in the [project report](./reports/final/main.pdf).

- `data/`: Contains files for generating the synthetic dataset (Section 4.2).
- `models/`: Contains the pre-trained models for text detection and recognition (Section 4.1).
- `notebooks/`: Contains the Jupyter notebooks for experiments and evaluation (Sections 5.1 to 5.3).
  - `confidence.ipynb`: Plot the confidence scores on the target dataset (Figures 5.7, 5.8).
  - `crnn_rec.ipynb`: Train the CRNN model for text recognition (Section 5.1.3).
  - `eval_comp.ipynb`: Compare the system performance with baselines (Figures 5.9 to 5.13).
  - `fnzz.ipynb`: Analyze the Funü Zazhi dataset (Figures 3.1, 3.2).
  - `frcnn_det.ipynb`: Train the Faster R-CNN model for text detection (Section 5.1.1).
  - `utils.ipynb`: Run utility functions for the experiments.
- `output/`: Contains the output files from the OCR models in experiments (Section 5.1).
  - `samples/`: Contains the OCR output of 20 test images.
  - `images/`, `positions/`, `raw/`, `text/`: Contains the image (with boxes) / positions / raw JSON data / text output from the OCR system on all target images (to be obtained by `predict.py`).
  - `confidence.csv`: Contains the confidence scores of the OCR system on all target images.
- `plots/`: Contains the plots generated in the experiments (Section 5.2).
- `reports/`: Contains the source code, figures, or PDFs for all project reports.
- `slides/`: Contains the source code, figures and PDF for presentation.
- `src/`: Contains the source code for the project.
  - `app.py`: Run the user interface of the OCR system (Section 4.3).
  - `compare.py`: Utility functions for comparing the system with baselines (Section 5.3).
  - `convert.py`: Convert the original text output from traditional Chinese to simplified Chinese.
  - `dataset.py`: Create an image dataset of 6,000 Chinese characters in 12 fonts.
  - `download.py`: Download the Funü Zazhi dataset, including images and the table of contents.
  - `evaluate_det.py`: Evaluate the text detection model and output Figures 5.2, 5.3.
  - `evaluate_rec.py`: Evaluate the text recognition model and output Figures 5.5, 5.6.
  - `hp_tuning.py`: Perform hyperparameter tuning for the system and output Figures 5.1, 5.4.
  - `model.py`: Define the deep learning models in the OCR system (Section 4.1).
  - `predict.py`: Perform OCR methods on images and save the output (Section A.3).
  - `sort.py`: Sort the output text data of Funü Zazhi by content type, publication date, and confidence scores and save to different folders (Section A.3).
  - `table.py`: Extract data from the table of contents downloaded by `download.py`, which contains the content type and title for each image, and save it to a CSV file (Section A.3).
  - `utils.py`: Utility functions for the system, such as data processing, image visualization, and evaluation metrics.
  - `easy_ocr.py`, `paddle_ocr.py`, `tesseract_ocr.py`: Perform EasyOCR / PaddleOCR / Tesseract on test images and save the output to `output/samples/`.
- `target/`: Contains images (to be downloaded) of Funü Zazhi, and annotated test images.
  - `samples/`: Contains 20 annotated test images for evaluation.
  - `tables/`: Contains the table of contents of Funü Zazhi (downloaded by `download.py`).
  - `data_raw.csv`, `data.csv`: The output of `table.py` containing content type and title for each image labeled as `yymm_xxxx` (Section 3.1).

## Installation
To run the OCR system, install Python and the relevant packages as follows:

```bash
# Clone the repository
git clone https://gitlab.doc.ic.ac.uk/ms423/msc-individual-project.git

# Navigate to the project directory
cd msc-individual-project

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate # Linux / MacOS
venv\Scripts\activate # Windows

# Install the required packages
pip install -r requirements.txt
```

The 20 test images are provided in the `target/samples/` directory. You can also download the whole Funü Zazhi dataset by:

```bash
python src/download.py -i
```

The downloaded images will be saved in the `target/images/` directory. The download may take several hours depending on network speed. The table of contents of Funü Zazhi is also provided in `target/data.csv`, which is used to sort the text content by type in `sort.py` (Section A.3).

## System Usage
### Text Detection and Recognition
To run the OCR system on the target images, use `predict.py` as follows:

```bash
# Perform OCR on the whole dataset (assumed downloaded)
python src/predict.py

# Perform OCR on the 20 test images
python src/predict.py -t

# Perform OCR on the whole dataset and overwrite existing output
python src/predict.py -o

# Perform OCR on the 20 test images and overwrite existing output
python src/predict.py -t -o
```

The output will be saved in the `output/` directory if using the whole dataset, or in `output/samples/` if using test images. By default, only JSON files with raw text detection and recognition results are saved. To include other output files, such as ordered text content, text line positions, or images with bounding boxes, set the corresponding flags in `predict.py`:

```python
output_raw = True  # Save raw JSON data
output_text = True  # Save ordered text content
output_positions = True  # Save positions of text lines
output_image = True  # Save image with bounding boxes
```

To convert traditional Chinese characters to simplified Chinese after obtaining the text content, run:

```bash
python src/convert.py
```

The converted text will be saved in `output/text_simplified/`. To sort the text content by content type, publication date and confidence score, run:

```bash
python src/sort.py
```

The sorted content will be saved in the subdirectories `text_sorted_type/`, `text_sorted_year/`, and `text_sorted_confidence/` under `output/` respectively.

To start the OCR system's user interface, run:

```bash
python src/app.py
```

And the user interface will display shortly. You may follow the buttons and checkboxes to upload images and view the detection and recognition results in different options. The detailed demonstration of the user interface is provided in Section 4.3.

### Training and Evaluation
To train the text detection and recognition models, you first need to create an image dataset of 6,000 Chinese characters in 12 fonts by:

```bash
python src/dataset.py
```

The dataset will be saved in `data/train_6k/`. You can then train text detection and recognition models by following the steps in `notebooks/frcnn_det.ipynb` and `notebooks/crnn_rec.ipynb` respectively.

To evaluate the text detection and recognition models and showing hyperparameter tuning results, run:

```bash
# Evaluate the text detection model
python src/evaluate_det.py

# Evaluate the text recognition model
python src/evaluate_rec.py

# Show the hyperparameter tuning results
python src/hp_tuning.py
```

The relevant plots will be saved in the `plots/` directory. To perform OCR using the baseline methods, run:

```bash
# Perform EasyOCR on test images
python src/easy_ocr.py

# Perform PaddleOCR on test images
python src/paddle_ocr.py

# Perform Tesseract on test images
python src/tesseract_ocr.py
```

The output will be saved in `output/samples/`. You can then compare the system with baselines in `notebooks/eval_comp.ipynb`. The results of these experiments are discussed in Sections 5.2 and 5.3.