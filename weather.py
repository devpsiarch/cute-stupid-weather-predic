pip install pandas

!pip install scikit-learn

import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# wll read the data here

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv',index_col=False)

#print(test_data)

train_x = train_data.iloc[:, :-1].values  # Features (all columns except the last one)
train_y = train_data.iloc[:, -1].values   # Labels (last column)

predic = test_data.iloc[:,1:].values

#print(train_x)
#print(train_y)
#print(predic)

# prep the taining
scaler = StandardScaler()
x_scaled = scaler.fit_transform(train_x)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(train_y)

predic_scaled = scaler.fit_transform(predic)

# ill split the data to check if this shit is working
X_train, X_val, y_train, y_val = train_test_split(x_scaled, y_encoded, test_size=0.2, random_state=52)

model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=52)

model.fit(X_train,y_train) # train ...

# test accuracy
val_accuracy = model.score(X_val, y_val)
print("Accuracy:", val_accuracy)

# Make predictions on the test data

test_predictions = model.predict(predic_scaled)
predictions_decoded = encoder.inverse_transform(test_predictions)
# Print or save the predictions
print("Test Data Predictions:", predictions_decoded)

# saving it to go
df = pd.DataFrame({
    'Index': np.arange(len(predictions_decoded)),  # Add an index if needed
    'Prediction': predictions_decoded
})

# Save to CSV
df.to_csv('predictions.csv', index=False)  # Set `index=True` if you want the DataFrame index saved

print("Predictions saved to 'predictions.csv'")
