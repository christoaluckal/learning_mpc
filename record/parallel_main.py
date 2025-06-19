from car import Car
from draw import Draw
from controllers import PID, MPC
import cv2
from utils import *
import argparse
from parameters import *
import time
from collections import deque
import pickle

data_queue = deque(maxlen=int(1e5))

parser = argparse.ArgumentParser(description='Process some integers.')


parser.add_argument('--controller', default="PID", 
		help="Select the controller for the robot.\nAvailable:\n1) MPC\n2) PID\nDefault: PID")
parser.add_argument('--proc_id', type=int, default=0)
parser.add_argument('--viz', action='store_true',default=False)

args = parser.parse_args()
controller_name = 'MPC'


if controller_name not in ["PID", "MPC"]:
	print("Invalid controller used. Available controllers: MPC and PID.")
	exit()

print(f"Using {controller_name} Controller.")

way_points = []
num_wps = 100
wps_x = np.random.randint(50, VIEW_W-50, (num_wps,1))
wps_y = np.random.randint(50, VIEW_W-50, (num_wps,1))
for w in range(wps_x.shape[0]):
	way_points.append([wps_x[w,0],wps_y[w,0]])
 
def add_waypoint(event, x, y, flags, param):
    global way_points
    if event == cv2.EVENT_LBUTTONDOWN:
        way_points.append([x, y])
    if event == cv2.EVENT_RBUTTONDOWN:
        way_points.pop()

if args.viz:
	draw = Draw(VIEW_W, VIEW_H, window_name = "Canvas", mouse_callback = add_waypoint)

car = Car(50, 50)

if controller_name == "PID":
	controller = PID(kp_linear = 0.5, kd_linear = 0.1, ki_linear = 0,
							kp_angular = 3, kd_angular = 0.1, ki_angular = 0)
if controller_name == "MPC":
	controller = MPC(horizon = MPC_HORIZON)


lw = 0
rw = 0
current_idx = 0
linear_v = 0
angular_v = 0
car_path_points = []
k = None
while True:
	if args.viz:
		draw.clear()
		draw.add_text("Press the right click to place a way point, press the left click to remove a way point", 
						color = (0, 0, 0), fontScale = 0.5, thickness = 1, org = (5, 20))
		if len(way_points)>0:
			draw.draw_path(way_points, color = (200, 200, 200), thickness = 1)

		if len(car_path_points)>0:
			draw.draw_path(car_path_points, color = (255, 0, 0), thickness = 1, dotted = True)

		draw.draw(car.get_points(), color = (255, 0, 0), thickness = 1)
	
	if args.viz:
		k = draw.show()
	# k = draw.show()

	x, _ = car.get_state()
	if len(way_points)>0 and current_idx != len(way_points):
		car_path_points.append([int(x[0, 0]), int(x[1, 0])])
		goal_pt = way_points[current_idx]

		if controller_name == "PID":
			linear_v, angular_v = controller.get_control_inputs(x, goal_pt, car.get_points()[2], current_idx)
		
		if controller_name == "MPC":
			px,pv = car.get_state()
			cx,cy,cyaw = px[0,0], px[1,0], px[2,0]
			cx = np.around(cx,2)
			cy = np.around(cy,2)
			cyaw = np.around(cyaw,2)
			cv,cw  = pv[0,0], pv[2,0]
			cv = np.around(cv,2)
			cw = np.around(cw,2)
			# print(f"Proc {args.proc_id} Current State: {cx}, {cy}, {cyaw}, {cv}, {cw}")
			linear_v, angular_v = controller.optimize(car = car, goal_x = goal_pt)
   
   
			linear_v += np.random.uniform(-2.5,2.5)
			angular_v += np.random.uniform(-0.3,0.3)
			# print(f"Optimized Control Inputs: {linear_v}, {angular_v}")
			# time.sleep(1)
		
		dist = get_distance(x[0, 0], x[1, 0], goal_pt[0], goal_pt[1])
		if dist<10:
			current_idx+= 1
	else:
		linear_v = 0
		angular_v = 0

	data_queue.append(car.last_row+[linear_v,angular_v])
	# print(data_queue[-1])
	car.update(linear_v, angular_v,DELTA_T)

	if len(data_queue) % 1000 == 0:
		print(f"Proc {args.proc_id}: {len(data_queue)}")
		with open(f'data_{args.proc_id}.pkl','wb') as f:
			pickle.dump(data_queue,f)
   
	if len(data_queue) == data_queue.maxlen:
		break

	if k == ord("q"):
		break