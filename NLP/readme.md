# NLP-Applied-to-the-Study-of-Women-s-Social-Status-in-the-May-Fourth-Movement

## Project Overview

This project explores the changes in women's social status and roles in China between 1915 and 1931, with a particular focus on the May Fourth Movement (1919) and the National Revolution (1926-1928). By analyzing historical text data from *The Ladies' Journal* using advanced Natural Language Processing (NLP) technologies, this study reveals shifts in women's roles across education, occupation, family, and marriage.

## Objectives

- **Investigate Changes**: Examine the evolution of women's social status and roles.
- **Apply Modern Techniques**: Utilize LDA topic modeling and DeBERTa sentiment analysis for data exploration.
- **Enhance Analysis Methods**: Address challenges in Chinese text segmentation and context understanding with innovative techniques.

## Dataset

The dataset originates from *The Ladies' Journal*, spanning from 1915 to 1931. The data was initially in PDF format and converted to text using OCR technology. We selected simplified text with a recognition accuracy of over 90% for analysis.

## Methodology

1. **Data Preprocessing**: Clean and segment text data using advanced tokenization techniques.
2. **Topic Modeling**: Apply LDA modeling to identify and analyze themes within the data.
3. **Sentiment Analysis**: Use the DeBERTa-v3-large model to evaluate sentiment and topic changes.
4. **Keyword Matching**: Introduce keyword matching technology to capture complex topic dynamics from a new perspective.
5. **Visualization**: Generate visualizations to present theme changes and sentiment trends.

## Key Findings

- **Economic and Social Status**: There is a significant increase in mentions of women's economic and social status, reflecting enhanced participation since the May Fourth Movement.
- **Family, Love, and Marriage**: The relevance of new-era keywords has increased, indicating a shift from traditional roles and a pursuit of personal fulfillment.
- **Consumerism**: Modern consumerism has greatly influenced women's lifestyles and consumption patterns.
- **Education and Culture**: Advances in education and culture have significantly improved women's social status and career opportunities.

## Contributions

- **Innovative Analysis Framework**: The integration of LDA topic modeling with the DeBERTa model for multi-dimensional topic trend analysis.
- **Enhanced Segmentation Technique**: Improved word segmentation for more accurate long-span text analysis.

## Installation

To run the code and reproduce the analysis results, install the necessary Python packages using:

```bash
pip install -r requirements.txt


```

## Project Structure

The project directory is organized as follows:

- **`requirements.txt`**: Contains the list of Python dependencies required for the project.
- **`src/`**: This directory includes the source code for data processing, topic modeling, and sentiment analysis.
- **`results/`**: Stores the processed data, topic modeling results, and output visualizations.
- **`config/`**: Includes configuration files for setting parameters and options used throughout the project.
- **`dataset/`**: Contains both the original and processed data used in the analysis.
