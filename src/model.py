from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_model(input_shape, learning_rate=0.0001):
    model = Sequential([
        LSTM(128, input_shape=input_shape),
        Dense(64, activation='relu'), 
        Dense(4, activation='softmax')  # Output layer with 4 units for 4 classes and softmax activation
    ])
    
    # Define the optimizer with a custom learning rate
    optimizer = Adam(learning_rate=learning_rate)
    
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
