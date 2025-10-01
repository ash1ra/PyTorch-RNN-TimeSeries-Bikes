# PyTorch RNN Time Series Forecasting for Bike Sharing

This project implements a recurrent neural network (LSTM) using PyTorch to forecast bike sharing demand from the [Bike Sharing Dataset](https://www.kaggle.com/datasets/marklvl/bike-sharing-dataset). The model predicts hourly bike rental counts in Washington D.C. based on temporal patterns, weather conditions, and historical data.

This project demonstrates time series forecasting with sequence-to-one prediction, where 96 hours of historical data are used to predict the next hour's bike rental count.

## Model Architecture

The neural network is an LSTM-based recurrent model:
- **Input Sequence**: 96 time steps (hours);
- **Categorical Features**: 5 features (year, holiday status, working day, season, weather) processed through embedding layers;
- **Numerical Features**: 12 features (cyclical time encodings, weather data, lag features);
- **Hidden Layer**: 64 LSTM units across 2 layers;
- **Output Layer**: 1 neuron (predicted bike count);
- **Loss Function**: RMSE (Root Mean Square Error);
- **Evaluation Metric**: R² score;
- **Optimizer**: Adam with weight decay (L2 regularization);
- **Learning Rate Scheduler**: ReduceLROnPlateau;
- **Regularization**: Dropout (0.3) and early stopping.

## Dataset

The [Bike Sharing Dataset](https://www.kaggle.com/datasets/marklvl/bike-sharing-dataset) contains hourly and daily bike rental data from Washington D.C.'s Capital Bikeshare system (2011-2012). The dataset includes:
- **Temporal features**: hour, day of week, month, year;
- **Weather data**: temperature, humidity, wind speed, weather conditions;
- **Contextual information**: season, holidays, working days;
- **Target variable**: hourly bike rental count.

### Data Preprocessing

- **Log transformation**: Applied to target variable for better distribution;
- **Standardization**: Z-score normalization for continuous features;
- **Cyclical encoding**: Sine/cosine transformation for periodic features (hour, month, day of week);
- **Lag features**: Historical counts at 1, 24, and 168 hours (1 hour, 1 day, 1 week);
- **Train/Val/Test split**: 70% / 15% / 15%.

## Project Structure

```
.
├── app.py                      # Main training script
├── model.py                    # LSTM model architecture
├── data_preprocessing.py       # Feature engineering and normalization
├── data_loading.py             # Dataset class and dataloaders
├── utils.py                    # Training loop, evaluation, plotting utilities
└── data/                       # Data directory
    ├── hour.csv                # Raw dataset (download from Kaggle)
    ├── processed_data.csv      # Preprocessed features
    ├── train_dataset/          # Training data
    ├── val_dataset/            # Validation data
    ├── test_dataset/           # Test data
    └── best_model.safetensors  # Saved model weights
```

## Configuration

Key hyperparameters in `app.py`:
- **Batch size**: 16
- **Hidden size**: 64 LSTM units
- **Number of layers**: 2
- **Dropout**: 0.3
- **Learning rate**: 0.001
- **Weight decay**: 1e-3
- **Max epochs**: 500
- **Early stopping patience**: 20 epochs
- **Sequence length**: 96 hours (defined in `data_loading.py`)

## Results

The model outputs:
- Training and validation loss curves;
- Predictions vs targets plots for train/val/test sets;
- RMSE and R² score metrics on the test set.

## Setting Up and Running the Project

1. Clone the repository:
```bash
git clone https://github.com/ash1ra/PyTorch-RNN-TimeSeries-Bikes.git
cd PyTorch-RNN-TimeSeries-Bikes
```

2. Create `.venv` and install dependencies:
```bash
uv sync
```

3. Activate a virtual environment:
```bash
# On Windows
.venv\Scripts\activate
# On Unix or MacOS
source .venv/bin/activate
```

4. Preprocess the data:
```bash
python data_preprocessing.py
```

5. Create train/validation/test datasets:
```bash
python data_loading.py
```

6. Train the model:
Customize hyperparameters in `app.py` if needed, then run:
```bash
python app.py
```
