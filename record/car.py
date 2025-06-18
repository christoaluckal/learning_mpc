import numpy as np
from parameters import *

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
  
		self.last_row = [self.x[0,0], self.x[1,0], self.x[2,0], self.x_dot[0,0], self.x_dot[2,0]]

		self.car_dims = np.array([
										[-self.b, -self.b, 1],
										[1e-6 		, -self.b, 1],
										[ self.b,  		1e-6, 1],
										[ 1e-6, 	   self.b, 1],
										[ -self.b, self.b, 1]
									])

		self.get_transformed_pts()


	def update(self, linear_v, angular_v, dt):
		self.x_dot = np.array([
										[linear_v],
										[1e-6],
										[angular_v]
									])
		ikine_mat = np.array([
							[1/self.r, 1e-6, self.b/self.r],
							[1/self.r, 1e-6, -self.b/self.r]
							])

		self.wheel_speed = ikine_mat@self.x_dot
		self.wheel_speed[self.wheel_speed>MAX_WHEEL_ROT_SPEED_RAD] = MAX_WHEEL_ROT_SPEED_RAD;
		self.wheel_speed[self.wheel_speed<MIN_WHEEL_ROT_SPEED_RAD] = MIN_WHEEL_ROT_SPEED_RAD;

		kine_mat = np.array([
							[self.r/2  		  , self.r/2],
							[1e-6 		 		  ,	1e-6],
							[self.r/(2*self.b), -self.r/(2*self.b)]
							])

		self.x_dot = kine_mat@self.wheel_speed
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
							[self.x_dot[0, 0]],
							[self.x_dot[2, 0]]
						])
		self.x = A@self.x + B@vel
  
		self.x[2,0] %= 2*np.pi
  
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

	# def inverse_kinematics(self):
	# 	ikine_mat = np.array([
	# 						[1/self.r, 0, self.b/self.r],
	# 						[1/self.r, 0, -self.b/self.r]
	# 						])

	# 	return ikine_mat@self.x_dot

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