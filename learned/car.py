import numpy as np
from parameters import *
import pickle
import torch
from torch import nn

class SimpleNN(nn.Module):
	def __init__(self, input_size, output_size):
		super(SimpleNN, self).__init__()
		self.fc1 = nn.Linear(input_size, 128)
		self.fc2 = nn.Linear(128, 128)
		self.fc3 = nn.Linear(128, output_size)
	def forward(self, x):
		x = torch.relu(self.fc1(x))
		x = torch.relu(self.fc2(x))
		x = self.fc3(x)
		return x

class Car:
	def __init__(self, x, y):
		self.x = np.array([
							[x],
							[y],
							[1e-6]
						  ])

		self.x_dot = np.array([
							[1e-6],
							[1e-6],
							[1e-6]
						  ])

		self.wheel_speed = np.array([
										[1e-6],
										[1e-6]
									])

		self.b = 25
		self.r = 5
  
		# self.last_row = [1e-6,1e-6,1e-6,1e-6,1e-6]
		self.last_row = [x,y,1e-6,1e-6,1e-6]

		self.car_dims = np.array([
										[-self.b, -self.b, 1],
										[1e-6 		, -self.b, 1],
										[ self.b,  		1e-6, 1],
										[ 1e-6, 	   self.b, 1],
										[ -self.b, self.b, 1]
									])

		self.get_transformed_pts()
		# self.model = None
		# with open('model.pkl', 'rb') as f:
		# 	self.model = pickle.load(f)
		with open('model.pkl','rb') as f:
			self.lin_reg_model = pickle.load(f)

		self.nn_model = SimpleNN(input_size=7, output_size=3)
		self.nn_model.load_state_dict(torch.load('nn_model.pth'))
		self.nn_model.eval()

		dummy_input = self.last_row + [1e-6, 1e-6]
		dummy_input = np.array([dummy_input])
		dummy_tensor = torch.tensor(dummy_input, dtype=torch.float32)
		with torch.no_grad():
			print(f"Dummy input: {dummy_input}")
			dummy_output = self.nn_model(dummy_tensor).detach().numpy()
			print(f"Dummy output: {dummy_output}")

		input()
		self.nn_model.eval()
		self.nn_model.to(torch.float32)

	def update(self, linear_v, angular_v, dt, from_model=False):
		# if not from_model:
		# 	self.x_dot = np.array([
		# 									[linear_v],
		# 									[1e-6],
		# 									[angular_v]
		# 								])
		# 	ikine_mat = np.array([
		# 						[1/self.r, 1e-6, self.b/self.r],
		# 						[1/self.r, 1e-6, -self.b/self.r]
		# 						])

		# 	self.wheel_speed = ikine_mat@self.x_dot
		# 	self.wheel_speed[self.wheel_speed>MAX_WHEEL_ROT_SPEED_RAD] = MAX_WHEEL_ROT_SPEED_RAD;
		# 	self.wheel_speed[self.wheel_speed<MIN_WHEEL_ROT_SPEED_RAD] = MIN_WHEEL_ROT_SPEED_RAD;

		# 	kine_mat = np.array([
		# 						[self.r/2  		  , self.r/2],
		# 						[1e-6 		 		  ,	1e-6],
		# 						[self.r/(2*self.b), -self.r/(2*self.b)]
		# 						])

		# 	self.x_dot = kine_mat@self.wheel_speed
		# 	A = np.array([
		# 					[1, 1e-6, 1e-6],
		# 					[1e-6, 1, 1e-6],
		# 					[1e-6, 1e-6, 1]
		# 				])
		# 	B = np.array([
		# 					[np.sin(self.x[2, 0] + np.pi/2)*dt,  1e-6],
		# 					[np.cos(self.x[2, 0] + np.pi/2)*dt,  1e-6],
		# 					[1e-6					 , dt]
		# 				])

		# 	vel = np.array([
		# 						[self.x_dot[0, 0]],
		# 						[self.x_dot[2, 0]]
		# 					])
		# 	self.x = A@self.x + B@vel
	
		# 	self.x[2,0] %= 2*np.pi
		# else:
		# 	# STATE
		# 	d_row = np.array([self.last_row + [linear_v, angular_v]])
		# 	d_row[:,[0,1]] /= 700
		# 	d_row[:,2] /= 2*np.pi
		# 	d_row[:,3] /= 12.5
		# 	d_row[:,4] /= 4.5
		# 	y = self.model.predict(d_row)
		# 	y[:,[0,1]] *= 700
		# 	y[:,2] *= 2*np.pi
		# 	# input()
		# 	self.x = np.array([
		# 		[y[0,0]],
		# 		[y[0,1]],
		# 		[y[0,2]%(2*np.pi)]
		# 	])

		# 	self.x_dot = np.array([
		# 		[linear_v],
		# 		[1e-6],
		# 		[angular_v]
		# 	])
		print(f"Input: {self.last_row + [linear_v, angular_v]}")
		x_dot = np.array([
								[linear_v],
								[1e-6],
								[angular_v]
							])
		ikine_mat = np.array([
							[1/self.r, 1e-6, self.b/self.r],
							[1/self.r, 1e-6, -self.b/self.r]
							])

		ws = ikine_mat@self.x_dot
		ws[ws>MAX_WHEEL_ROT_SPEED_RAD] = MAX_WHEEL_ROT_SPEED_RAD;
		ws[ws<MIN_WHEEL_ROT_SPEED_RAD] = MIN_WHEEL_ROT_SPEED_RAD;

		kine_mat = np.array([
							[self.r/2  		  , self.r/2],
							[1e-6 		 		  ,	1e-6],
							[self.r/(2*self.b), -self.r/(2*self.b)]
							])

		x_dot = kine_mat@ws
		A = np.array([
						[1, 1e-6, 1e-6],
						[1e-6, 1, 1e-6],
						[1e-6, 1e-6, 1]
					])
		B = np.array([
						[np.sin(self.x[2, 0] + np.pi/2)*dt,  1e-6],
						[np.cos(self.x[2, 0] + np.pi/2)*dt,  1e-6],
						[1e-6					 , dt]
					])

		vel = np.array([
							[x_dot[0, 0]],
							[x_dot[2, 0]]
						])
		real_x = A@self.x + B@vel
		real_x[2,0] %= 2*np.pi


		input_row = np.array([self.last_row + [linear_v, angular_v]])
		input_row = np.around(input_row, decimals=3)

		y = self.lin_reg_model.predict(input_row)

		lr_x = np.array([
			[y[0,0]],
			[y[0,1]],
			[y[0,2]%(2*np.pi)]
		])
		
		nn_input = torch.tensor(input_row, dtype=torch.float32)
		nn_x = self.nn_model(nn_input).detach().numpy()

		nn_x = np.array([
			[nn_x[0,0]],
			[nn_x[0,1]],
			[nn_x[0,2]%(2*np.pi)]
		])
		
		self.x = real_x
		if not (self.lin_reg_model and self.nn_model):
			pass
		else:
			print(f"Real x: {real_x.flatten()}")
			print(f"LinReg x: {lr_x.flatten()}")
			print(f"NN x: {nn_x.flatten()}")
			print()
			# self.x = nn_x

		self.x_dot = np.array([
								[linear_v],
								[1e-6],
								[angular_v]
							])
		self.last_row = [self.x[0,0],self.x[1,0],self.x[2,0],self.x_dot[0,0],self.x_dot[2,0]]


	def get_state(self):
		return self.x, self.x_dot

	def forward_kinematics(self):
		kine_mat = np.array([
							[self.r/2  		  , self.r/2],
							[1e-6 		 		  ,	1e-6],
							[self.r/(2*self.b), -self.r/(2*self.b)]
							])

		return kine_mat@self.wheel_speed

	def get_transformed_pts(self):
		rot_mat = np.array([
							[ np.cos(self.x[2, 0]), np.sin(self.x[2, 0]), self.x[0, 0]],
							[-np.sin(self.x[2, 0]), np.cos(self.x[2, 0]), self.x[1, 0]],
							[1e-6, 1e-6, 1]
							])

		self.car_points = self.car_dims@rot_mat.T

		self.car_points = self.car_points.astype("int")

	def get_points(self):
		self.get_transformed_pts()
		return self.car_points