# Eye-CNN-ViT-Flask: Diabetic Retinopathy Web App 🩺

This is a Flask web application that provides an interactive interface for classifying diabetic retinopathy using pre-trained deep learning models. Upload a retinal fundus image and get instant, side-by-side predictions from a Convolutional Neural Network (ResNet50) and a Vision Transformer (ViT).

The application performs a dual-output classification, assessing both the severity of retinopathy and the risk of macular edema.

![App Screenshot](https://i.imgur.com/L2f7JjA.png)

***

## Features

-   **Interactive Web Interface:** A simple and fun user interface for uploading retinal images and viewing results. 🖼️
-   **Model Showdown:** Compares predictions from a battle-tested **CNN (ResNet50)** and a modern **Vision Transformer (ViT)**. 🥊
-   **Dual-Output Classification:** For each image, the app predicts two critical conditions:
    -   **Retinopathy Grade:** A 5-level severity scale (No DR, Mild, Moderate, Severe, Proliferative DR).
    -   **Macular Edema Risk:** A 3-level risk scale (No DME, Grade 1, Grade 2).
-   **Instant Feedback:** Get predictions immediately after uploading an image.

***

## Project Structure

The repository is organized as follows:


Eye-CNN-ViT-Flask/
├── app.py              # Main Flask application
├── models/             # Pre-trained models and architecture
│   ├── architecture.py
│   ├── cnn_model.pth     # <-- Trained ResNet50 model weights
│   └── vit_model.pth     # <-- Trained ViT model weights
├── static/             # CSS, JavaScript, images, and sounds
├── templates/          # HTML templates
│   └── index.html
└── utils/              # Image preprocessing utilities


***

## Setup & Installation

Follow these steps to get the application running on your local machine.

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/](https://github.com/)<your-username>/Eye-CNN-ViT-Flask.git
    cd Eye-CNN-ViT-Flask
    ```

2.  **Install Dependencies**
    It's recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
    **Note:** The pre-trained model weights (`cnn_model.pth` and `vit_model.pth`) are required and should be located in the `models/` directory.

***

## How to Run

1.  **Start the Flask Server**
    From the root directory of the project, run:
    ```bash
    python app.py
    ```

2.  **Access the Application**
    Open your web browser and navigate to:
    [**http://127.0.0.1:5000**](http://127.0.0.1:5000)

3.  **Use the App**
    Simply click the upload area, select a retinal fundus image, and the models will automatically provide their classifications.

***

## Acknowledgments

-   The models were trained on the **IDRiD (Indian Diabetic Retinopathy Image Dataset)**.
-   Built with **PyTorch** and **Flask**.
-   The Vision Transformer implementation utilizes the **`timm`** library.

***

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
