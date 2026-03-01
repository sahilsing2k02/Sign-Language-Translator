# Realtime Sign Language Translation

A real-time application for detecting and recognizing sign language gestures using a webcam feed.

The Sign Language Recognition project focuses on developing an intelligent system that can understand and interpret hand gestures used in sign language and convert them into text or speech. The main goal of this project is to help bridge the communication gap between hearing-impaired individuals and the general public.

In this project, we used Machine Learning and Deep Learning techniques, especially Convolutional Neural Networks (CNNs), to recognize hand gestures from images or live video input. The system captures hand gestures through a camera, processes the image, extracts important features, and predicts the corresponding alphabet, word, or sentence.

The dataset consists of labeled images of different sign language gestures. After preprocessing and training the model, it can accurately classify gestures in real time. Technologies such as Python, OpenCV, TensorFlow/Keras were used for implementation.

This project demonstrates practical applications of AI in assistive technology and highlights how deep learning can be used to solve real-world social problems.

## Table of Contents

- [Aim](#aim)
- [Project Overview](#project-overview)
- [Demo](#demo)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Project Report](#project-report)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Aim

This project aims to create a sign language translator using machine learning techniques and Python programming. The application utilizes various modules, primarily Mediapipe, Landmark, and Random Forest algorithms to interpret and translate sign language gestures into text or spoken language.

## Project Overview

Sign language is a crucial form of communication for individuals with hearing impairments. This project focuses on bridging the communication gap by creating a tool that can interpret sign language gestures in real-time and convert them into understandable text or speech.

This project leverages Flask for the web interface and TensorFlow/Keras for the machine learning model to recognize sign language gestures in real-time from a webcam feed.

<img src="hand-signs-of-the-ASL-Language.png"  width="60%"/>

> American Sign Language Convention for Alphabets.

<img src="sign%20language%202.jpg"  width="60%"/>

> Custom Sign Language for Words / Sentences.




## Features
* **Real-time sign language recognition**: Captures hand gestures using the Mediapipe library to track landmarks and movements.
* **Landmark analysis**: Utilizes Landmark module to extract key points and gestures from hand movements.
* **Machine learning translation**: Employs Random Forest algorithm to classify and interpret gestures into corresponding text.
* **Text-to-speech**: For better communication the text can be converted to spoken language using the speech synthesis.

## Getting Started
To get started with the Sign Language Translator, follow these steps:

### Prerequisites

1. **Python**: Provides a vast array of libraries and frameworks for machine learning, computer vision, and data processing.
2. **TensorFlow**: For building and training machine learning models.
3. **Scikit-learn**: For implementing the Random Forest algorithm for sign language recognition.
4. **Numpy**: For numerical computations and data manipulation.
5. **Mediapipe**: For real-time hand tracking and landmark detection.
6. **OpenCV**: For video processing and computer vision tasks.
7. **Flask**: Web framework to develop the application.
8. **Flask-SocketIO**: Adds low-latency bi-directional communication between clients and the server to Flask applications.

### Installation

1. Clone the repository:

```shell
cd sign2text
```

2. Create and activate a virtual environment:

  ```shell
  python -m venv venv
  ```
  ```shell
  source venv/bin/activate  # On Windows use `venv\Scripts\activate`
  ```

3. Install required libraries:

  ```shell
  pip install -r requirements.txt
  ```

4. Ensure a webcam is connected to your system.

## Usage

1. Start the Flask application:
    ```bash
    python app.py
    ```

2. Open your web browser and navigate to :
   ```bash
    http://127.0.0.1:5000/
    ```

4. The web interface will display the webcam feed and detected sign language gestures.

Some Samples of output
<img width="1908" height="978" alt="image" src="https://github.com/user-attachments/assets/27062d36-da59-43c1-b278-993697cffba1" />
<img width="1910" height="972" alt="Screenshot 2026-03-01 093224" src="https://github.com/user-attachments/assets/c8c37339-35cd-409f-97fe-05d1aa08d052" />
<img width="1913" height="979" alt="Screenshot 2026-03-01 093148" src="https://github.com/user-attachments/assets/995b0f0e-206b-4beb-aa3f-a7a13e7f245f" />


