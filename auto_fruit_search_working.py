# M4 - Autonomous fruit searching

# basic python packages
from ast import Global
import sys, os
import cv2
import numpy as np
import json
import argparse
import time
import re
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
# from machinevisiontoolbox import Image, CentralCamera

# import SLAM components
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco
import pygame                       # python package for GUI
import shutil                       # python package for file operations
import util.DatasetHandler as dh    # save/load functions

# import utility functions
sys.path.insert(0, "util")
from pibot import PenguinPi
import measure as measure

from YOLO.detector import Detector


x=None
y=None
robot_marker = None
global robot_pose
robot_pose = [0, 0, 0] 

aruco_marker_radius = 0.5
class Operate:
    def __init__(self, args):
        self.folder = 'pibot_dataset/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        else:
            shutil.rmtree(self.folder)
            os.makedirs(self.folder)

        # initialise data parameters
        if args.play_data:
            self.pibot = dh.DatasetPlayer("record")
        else:
            self.pibot = PenguinPi(args.ip, args.port)

        # initialise SLAM parameters
        self.ekf = self.init_ekf(args.calib_dir, args.ip)
        self.aruco_det = aruco.aruco_detector(
            self.ekf.robot, marker_length=0.07)  # size of the ARUCO markers

        if args.save_data:
            self.data = dh.DatasetWriter('record')
        else:
            self.data = None
        self.output = dh.OutputWriter('lab_output')
        self.command = {'motion': [0, 0],#[forward/down, left/right]
                        'inference': False,
                        'output': False,
                        'save_inference': False,
                        'save_image': False}
        self.quit = False
        self.pred_fname = ''
        self.request_recover_robot = False
        self.file_output = None
        self.ekf_on = False
        self.double_reset_comfirm = 0
        self.image_id = 0
        self.notification = 'Press ENTER to start SLAM'
        self.pred_notifier = False
        # a 5min timer
        self.count_down = 300
        self.start_time = time.time()
        self.control_clock = time.time()
        # initialise images
        self.img = np.zeros([240, 320, 3], dtype=np.uint8)
        self.aruco_img = np.zeros([240, 320, 3], dtype=np.uint8)
        self.detector_output = np.zeros([240, 320], dtype=np.uint8)
        if args.yolo_model == "":
            self.detector = None
            self.yolo_vis = cv2.imread('pics/8bit/detector_splash.png')
        else:
            self.detector = Detector(args.yolo_model)
            self.yolo_vis = np.ones((240, 320, 3)) * 100
        self.bg = pygame.image.load('pics/gui_mask.jpg')
        
    def update_slam(self, drive_meas):
        self.take_pic()
        lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)
        is_success = False
        if self.request_recover_robot:
            is_success = self.ekf.recover_from_pause(lms)
        if is_success:
            self.robot_pose = self.ekf.robot.state[:3]
        else:
            pose_predict = self.ekf.predict(drive_meas)
            self.ekf.add_landmarks(lms)
            pose_update = self.ekf.update(lms)
            if lms:
                self.robot_pose = pose_update
            else:
                self.robot_pose = pose_predict
        print('state: update_slam', self.robot_pose, lms)
        return self.robot_pose

    def init_ekf(self, datadir, ip):
        fileK = "{}intrinsic.txt".format(datadir)
        camera_matrix = np.loadtxt(fileK, delimiter=',')
        fileD = "{}distCoeffs.txt".format(datadir)
        dist_coeffs = np.loadtxt(fileD, delimiter=',')
        fileS = "{}scale.txt".format(datadir)
        scale = np.loadtxt(fileS, delimiter=',')
        if ip == 'localhost':
            scale /= 2
        fileB = "{}baseline.txt".format(datadir)
        baseline = np.loadtxt(fileB, delimiter=',') #this is the turn vel in 
        robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
        return EKF(robot)
    
    def detect_target(self):
        if self.command['inference'] and self.detector is not None:
            self.detector_output, self.network_vis = self.detector.detect_single_image(self.img)
            self.command['inference'] = False
            self.file_output = (self.detector_output, self.ekf)
            self.notification = f'{len(np.unique(self.detector_output))-1} target type(s) detected'
    
    def control(self):
        if args.play_data:
            lv, rv = self.pibot.set_velocity()
        else:
            lv, rv = self.pibot.set_velocity(
                self.command['motion'], tick=40, turning_tick=10)
        if not self.data is None:
            self.data.write_keyboard(lv, rv)
        dt = time.time() - self.control_clock
        drive_meas = measure.Drive(lv, rv, dt)
        self.control_clock = time.time()
        return drive_meas
    
    def get_robot_pose(self):
      robot_pose = self.ekf.robot.state.flatten()  
      return robot_pose

    
    
    
    
def get_EKF_SLAM(ip_address):
    # Extract the calibration parameters
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    if ip_address != '192.168.50.1':
        scale /= 2

    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')

    fileK = "calibration/param/intrinsic.txt"
    intrinsics = np.loadtxt(fileK, delimiter=',')

    fileD = "calibration/param/distCoeffs.txt"
    dist_coeffs = np.loadtxt(fileD, delimiter=',')

    # Initiialise the Robot used for EKF
    robot = Robot(baseline, scale, intrinsics, dist_coeffs)

    # Initialise the EKF SLAM Class
    ekf = EKF(robot)

    return ekf   
    
 
def get_robot_pose() -> np.ndarray:
    ####################################################
    # TODO: replace with your codes to estimate the pose of the robot
    # We STRONGLY RECOMMEND you to use your SLAM code from M2 here
    #    state_vector = ekf.get_state_vector()
    #    print(state_vector)

    # update the robot pose [x,y,theta]
    robot_pose = ekf.get_state_vector()[0:3, :].flatten()
    # Assuming get_state_vector() returns [x, y, theta]
    ####################################################
    while robot_pose[2] < 0 or robot_pose[2] > 2*np.pi:
        if robot_pose[2] < 0:
            robot_pose[2] += 2*np.pi
        else:
            robot_pose[2] -= 2*np.pi
    return robot_pose

    
    
def init_ekf(self, datadir, ip):
        fileK = "{}intrinsic.txt".format(datadir)
        camera_matrix = np.loadtxt(fileK, delimiter=',')
        fileD = "{}distCoeffs.txt".format(datadir)
        dist_coeffs = np.loadtxt(fileD, delimiter=',')
        fileS = "{}scale.txt".format(datadir)
        scale = np.loadtxt(fileS, delimiter=',')
        if ip == 'localhost':
            scale /= 2
        fileB = "{}baseline.txt".format(datadir)
        baseline = np.loadtxt(fileB, delimiter=',')
        robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
        return EKF(robot)


   
def drive_to_point(waypoint):
    robot_pose = get_robot_pose()
    #get pose 
    #robot_pose = get_robot_pose()
    # imports camera / wheel calibration parameters 
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')

    # Extract x and y coordinates from the waypoint
    x_target, y_target = waypoint
    #  print('pose', robot_pose, waypoint)


    # Calculate the angle (theta) to the target waypoint
    del_x = x_target - robot_pose[0]
    del_y = y_target - robot_pose[1]
    print("your way point: ",waypoint)
    print("Robot starting pose: ",robot_pose)
    distance = np.sqrt(del_x**2 + del_y**2)
    ppi.set_velocity([0,0])
    #for the 90 rot test cases
    if del_y==0:
        if del_x>0:
            theta=1.57
            theta_deg=np.degrees(theta)
        if del_x<0:
            theta=-1.57
            theta_deg=np.degrees(theta)
    elif del_x==0:
        theta=0
        theta_deg=np.degrees(theta)
        
    else:
        theta = np.arctan2(del_x, del_y)
        print("Theta to turn: ",theta)
    #clamp dat bitch
    while theta < 0:
        theta += 2*np.pi
    wheel_vel_turn = 20*operate.scale # tick*(m/s)/tick
    wheel_vel_drive = 50*operate.scale # tick*(m/s)/tick
    #drive time = 8.11s for 1m
    #ratios ma boi
    # turn towards the waypoint
    # TODO: turn_time= 9s
    angle_difference = theta - robot_pose[2]
    print("angle difference: ",angle_difference)
    #clamping dat bitch
    if angle_difference < 0:
        angle_difference += 2*np.pi
    elif angle_difference > 2*np.pi:
        angle_difference -= 2*np.pi
    print("angle difference after clamping: ",angle_difference)
    #CLAMP DAT BITCH
    if angle_difference > np.pi:
        turn_time = float(baseline*np.abs(2*np.pi-angle_difference)/(wheel_vel_turn*2)) # replace with your calculation
        print("turning right")
        drive_meas = operate.control([0, -1], turn_time)
    else:
        turn_time = float(baseline*np.abs(angle_difference)/(wheel_vel_turn*2)) # replace with your calculation
        print("turning left")
        drive_meas = operate.control([0, 1], turn_time)

    print("turn time: ", turn_time)

    #update the slam like in function
    operate.take_pic
    operate.update_slam(drive_meas)
    operate.control([0, 0], 0.01)
    robot_pose = get_robot_pose()


    #find robot pose again to account for theta
    robot_pose = get_robot_pose()
    # after turning, drive straight to the waypoint
    # TODO:drive_time=[
    if distance ==0:
        drive_time = 0
        print("Drive time: ",drive_time)
        #ppi.set_velocity([0, 0], tick=wheel_vel_drive, time=drive_time)
        drive_meas = operate.control([1,0], drive_time)
    else:
        drive_time = abs(distance/(wheel_vel))
        print("Drive Time: ",drive_time)
        #ppi.set_velocity([1, 0], tick= wheel_vel_drive/operate.scale, time=drive_time)
        drive_meas = operate.control([1,0], drive_time)

    #Update slam
    operate.take_pic()
    operate.update_slam(drive_meas)
    
    #update position
    robot_pose  = get_robot_pose()

    print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))

    return [waypoint[0],waypoint[1],0]
        


def read_true_map(fname):
    """Read the ground truth map and output the pose of the ArUco markers and 5 target fruits&vegs to search for

    @param fname: filename of the map
    @return:
        1) list of targets, e.g. ['lemon', 'tomato', 'garlic']
        2) locations of the targets, [[x1, y1], ..... [xn, yn]]
        3) locations of ArUco markers in order, i.e. pos[9, :] = position of the aruco10_0 marker
    """
    with open(fname, 'r') as fd:
        gt_dict = json.load(fd)
        fruit_list = []
        fruit_true_pos = []
        aruco_true_pos = np.empty([10, 2])

        # remove unique id of targets of the same type
        for key in gt_dict:
            x = np.round(gt_dict[key]['x'], 1)
            y = np.round(gt_dict[key]['y'], 1)

            if key.startswith('aruco'):
                if key.startswith('aruco10'):
                    aruco_true_pos[9][0] = x
                    aruco_true_pos[9][1] = y
                else:
                    marker_id = int(key[5]) - 1
                    aruco_true_pos[marker_id][0] = x
                    aruco_true_pos[marker_id][1] = y
            else:
                fruit_list.append(key[:-2])
                if len(fruit_true_pos) == 0:
                    fruit_true_pos = np.array([[x, y]])
                else:
                    fruit_true_pos = np.append(fruit_true_pos, [[x, y]], axis=0)

        return fruit_list, fruit_true_pos, aruco_true_pos
    
def is_collision_free(self, point):
    # Check if point collides with any rectangle obstacles
    for obstacle in self.obstacles:
        x, y, w, h = obstacle
        if (x <= point[0] <= x + w) and (y <= point[1] <= y + h):
            return False
    return True



def read_search_list():
    """Read the search order of the target fruits

    @return: search order of the target fruits
    """
    search_list = []
    with open('search_list.txt', 'r') as fd:
        fruits = fd.readlines()

        for fruit in fruits:
            search_list.append(fruit.strip())

    return search_list


def print_target_fruits_pos(search_list, fruit_list, fruit_true_pos):
    """Print out the target fruits' pos in the search order

    @param search_list: search order of the fruits
    @param fruit_list: list of target fruits
    @param fruit_true_pos: positions of the target fruits
    """

    print("Search order:")
    n_fruit = 1
    for fruit in search_list:
        for i in range(len(fruit_list)): # there are 5 targets amongst 10 objects
            if fruit == fruit_list[i]:
                print('{}) {} at [{}, {}]'.format(n_fruit,
                                                  fruit,
                                                  np.round(fruit_true_pos[i][0], 1),
                                                  np.round(fruit_true_pos[i][1], 1)))
        n_fruit += 1


def draw_map(self, waypoints, obstacles):
    # Setting up the plots
    fig = plt.figure(figsize=(6,6))
    ax = plt.gca()
    ax.cla() # clear things for fresh plot
    
    # change default range so that new circles will work
    ax.set_xlim((-1.5, 1.5))
    ax.set_ylim((-1.5, 1.5))

    ax.invert_xaxis()
    ax.invert_yaxis()
    
    circles = []
    for obj in obstacles:
        circles.append(plt.Circle(obj.center, obj.radius))

    for idx, waypoint in enumerate(waypoints):
        circles.append(plt.Circle((waypoint.x, waypoint.y), .03))
        if idx > 0:
            p_waypoint = waypoints[idx-1]
            ax.plot((p_waypoint.x, waypoint.x), (p_waypoint.y, waypoint.y), 'k-')

    for c in circles:
        ax.add_patch(c)

    fig.axes.append(ax)
    plt.show()
    
def on_click(event):
    global x, y, robot_pose, robot_marker
    if event.xdata is not None and event.ydata is not None:
        x, y = event.xdata, event.ydata
        plt.plot(x, y, 'ro')  # Display waypoint on map

        # Estimate the robot's pose and drive to the selected waypoint
        waypoint = [x, y]
        
    
        robot_pose=drive_to_point(waypoint)
        ax.plot([robot_pose[0], x], [robot_pose[1], y], 'k--')
        plt.draw()
        print("Robot ending pose: ",robot_pose)
        print("robot pose x rounded to 4 dec: ",round(robot_pose[0],4))
        print("waypoint x rounded to 4 dec: ",round(x,4))
        print("current angle: ", robot_pose[2])


        # Before plotting the new robot position, remove the previous one
        if robot_marker:
            robot_marker.remove()

        # Update the robot position on the map
        robot_marker, = ax.plot(robot_pose[0], robot_pose[1], 'bo', markersize=10)  # Note the comma after robot_marker, this is used to unpack the result from ax.plot

        if round(robot_pose[0],4) == round(x,4) and round(robot_pose[1],4) == round(y,4):
            return [waypoint[0], waypoint[1], robot_pose[2]]
        else:
            print("gg")
    else:
        print("Invalid click. Please select a valid point on the map.")


def update_slam(self, drive_meas):
    lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)
    if self.request_recover_robot:
        is_success = self.ekf.recover_from_pause(lms)
        if is_success:
            self.notification = 'Robot pose is successfully recovered'
            self.ekf_on = True
        else:
            self.notification = 'Recover failed, need >2 landmarks!'
            self.ekf_on = False
        self.request_recover_robot = False
    elif self.ekf_on:
        self.ekf.predict(drive_meas)
        self.ekf.add_landmarks(lms)
        self.ekf.update(lms)

def on_key(event):
    global robot_pose
    if event.key == 'r' or event.key == 'R':  # Check if the key pressed is 'R' or 'r'
        robot_pose = [0, 0, 0]
        print("Robot pose reset to [0, 0, 0]!")




# main loop
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Fruit searching")
    # CHANGE THIS LINE TO THE SLAM MAP
    parser.add_argument("--map", type=str, default='slam.txt') # change to 'M4_true_map_part.txt' for lv2&3
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    args, _ = parser.parse_known_args()

    ppi = PenguinPi(args.ip,args.port)
    ekf = get_EKF_SLAM(args.ip)
    # aruco_det = LoadMap(ekf.robot, marker_length=0.07)
    # vision_det = Detector(ip=args.ip)




    # read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    aruco_obstacles = [(pos[0]-aruco_marker_radius, pos[1]-aruco_marker_radius, 2*aruco_marker_radius, 2*aruco_marker_radius) for pos in aruco_true_pos]

    waypoint = [0.0,0.0]

    # measurements = aruco_det.load_landmark(aruco_true_pos)
    # ekf.init_lm_cov = 1e-3
    # ekf.add_landmarks(measurements)
    # ekf.update(measurements)
    # ekf.init_lm_cov = 1e1

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--yolo_model", default='YOLO/model/yolov8_model.pt')
    args, _ = parser.parse_known_args()
    operate = Operate(args)

    ekf = operate.init_ekf(args.calib_dir, args.ip)
    # operate.ekf.reset()

     
        # The following is only a skeleton code for semi-auto navigation
    while True:

         
        fig, ax = plt.subplots()
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.invert_yaxis()

        # Plot ArUco markers
        for pos in aruco_true_pos:
           ax.plot(pos[0], pos[1], 'bs', markersize=10)

         # Plot fruits and label them
        for i, pos in enumerate(fruits_true_pos):
          if fruits_list[i] in search_list:
            ax.plot(pos[0], pos[1], 'ro', markersize=10)  # Plot in red
          else:
             ax.plot(pos[0], pos[1], 'go', markersize=10)
          ax.text(pos[0], pos[1], fruits_list[i], fontsize=8, ha='right', va='top')
        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.mpl_connect('key_press_event', on_key)
        
        plt.show()
        



        # ekf= operate.init_ekf(args.calib_dir, args.ip)

        # drive_meas = operate.control()
        # operate.update_slam(drive_meas)

        # robot_pose= operate.robot_pose
        print("Robot Pose:", robot_pose)

        print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint, robot_pose))



        # ax.plot(robot_pose[0], robot_pose[1], 'bo', markersize=10)  # Assuming robot_pose is [x, y, theta]


        # Bind the figure to the waypoint selection function
    
        time.sleep(0.1)
        




    


