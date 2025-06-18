import pickle
import numpy as np
from sklearn.model_selection import train_test_split
np.set_printoptions(suppress=True,precision=3)
n = 12
fin_data = []
for i in range(n):
    try:
        with open(f'data_{i}.pkl','rb') as f:
            fin_data.extend(pickle.load(f))
    except:
        print(f"Error loading proc:",i)
    
    
data_array = np.array(fin_data)


data_array[:,[0,1]] /= 700
data_array[:,2] /= 2*np.pi
data_array[:,[3,5]] /= 12.5
data_array[:,[4,6]] /= 4.5


state_idx = [0,1,2,3,4]
control_idx = [5,6]
pred_idx = [0,1,2]

ith_row = data_array[:-1,state_idx+control_idx]
next_row = data_array[1:,pred_idx]

X = ith_row
y = next_row

X = np.around(X,decimals=3)
y = np.around(y,decimals=3)

print(X.shape,y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size=0.2,shuffle=True)
print(X_train.shape, Y_train.shape)
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train,Y_train)

from sklearn.metrics import r2_score, mean_squared_error

preds = model.predict(X_test)

print(f"R2:",r2_score(Y_test,preds))
print(f"MSE:",mean_squared_error(Y_test,preds))

import pickle
with open('../learned/model.pkl','wb') as f:
    pickle.dump(model,f)

random_idxs = np.random.randint(0,len(Y_test),size=(20))

random_ins = X_test[random_idxs]
random_outs = Y_test[random_idxs]
preds = model.predict(random_ins)

for x,y,z in zip(random_ins,random_outs,preds):
    print(f"In:{x}, True:{y}, Pred:{z} Error:{np.linalg.norm(y-z)}")
    # print(y-z)

import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def train_nn(X_train, Y_train, X_test, Y_test, epochs=1000, batch_size=32):
    input_size = X_train.shape[1]
    output_size = Y_train.shape[1]
    
    model = SimpleNN(input_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train_tensor)
        loss = criterion(outputs, Y_train_tensor)
        
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)
        test_outputs = model(X_test_tensor)
        test_loss = criterion(test_outputs, Y_test_tensor)
        print(f'Test Loss: {test_loss.item():.4f}')
    
    return model

# Train the neural network
nn_model = train_nn(X_train, Y_train, X_test, Y_test, epochs=1000, batch_size=32)

# Test the neural network
nn_preds = nn_model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy()
print(f"NN R2:", r2_score(Y_test, nn_preds))
print(f"NN MSE:", mean_squared_error(Y_test, nn_preds))

for x, y, z in zip(X_test[random_idxs], Y_test[random_idxs], nn_preds[random_idxs]):
    print(f"In:{x}, True:{y}, Pred:{z} Error:{np.linalg.norm(y-z)}")
    # print(y-z)