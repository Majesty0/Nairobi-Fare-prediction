# Nairobi Fare Prediction

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-black?logo=flask)](https://flask.palletsprojects.com/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-visualization-3f4f75?logo=plotly)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Nairobi Fare Prediction is a transport price prediction project built around a trained machine learning model, a Jupyter notebook for data exploration, and a Flask web app for interactive fare estimates.

The project uses trip and weather features such as rain, humidity, temperature, wind, distance, cab type, and destination to estimate the expected fare in Kenyan shillings.

## Features

- Web form for fare prediction with a clean Flask interface
- Notebook-based model exploration and feature engineering
- Pre-trained model, encoder, and imputer loaded from `.pkl` files
- Visual exploration with scatter matrix plots
- Responsive UI with a banner image and polished layout

## Project Structure

- `app.py` - Flask app for fare prediction
- `templates/index.html` - web UI template
- `static/css/style.css` - app styling
- `static/images/nairobi_matatu.jpg` - header image used by the app
- `Transportation_price_prediction.ipynb` - notebook for model exploration and training
- `model.pkl`, `encoder.pkl`, `imputer.pkl`, `transport_fare_model.pkl` - saved artifacts

## Setup

Install the required packages:

```bash
pip install -r requirements.txt
```

## Run the Web App

```bash
python app.py
```

Open the local URL printed in the terminal.

## Run the Notebook

Open `Transportation_price_prediction.ipynb` in Jupyter or VS Code and run the cells from top to bottom.

> **Note:** The notebook expects a local CSV file matching `Nairobi_Transport_Data*.csv` in the same folder.

> **Important:** The Flask app depends on the saved model artifacts. If any `.pkl` file is missing, predictions will fail at startup.

> **Warning:** Run the notebook from the repository folder so the file glob can find the dataset.

## Dependencies

See `requirements.txt` for the full list of Python packages used by the notebook and the web app.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
