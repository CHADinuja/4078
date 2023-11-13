# estimate the pose of target objects detected
import numpy as np
import json
import os
import ast
import cv2
from YOLO.detector import Detector
from sklearn.cluster import KMeans

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# list of target fruits and vegs types
# Make sure the names are the same as the ones used in your YOLO model
TARGET_TYPES = ['Orange', 'Lemon', 'Lime', 'tomato', 'Capsicum', 'Potato', 'Pumpkin', 'Garlic']


def estimate_pose(camera_matrix, obj_info, robot_pose):
    """
    function:
        estimate the pose of a target based on size and location of its bounding box and the corresponding robot pose
    input:
        camera_matrix: list, the intrinsic matrix computed from camera calibration (read from 'param/intrinsic.txt')
            |f_x, s,   c_x|
            |0,   f_y, c_y|
            |0,   0,   1  |
            (f_x, f_y): focal length in pixels
            (c_x, c_y): optical centre in pixels
            s: skew coefficient (should be 0 for PenguinPi)
        obj_info: list, an individual bounding box in an image (generated by get_bounding_box, [label,[x,y,width,height]])
        robot_pose: list, pose of robot corresponding to the image (read from 'lab_output/images.txt', [x,y,theta])
    output:
        target_pose: dict, prediction of target pose
    """
    # read in camera matrix (from camera calibration results)
    focal_length = camera_matrix[0][0]

    # there are 8 possible types of fruits and vegs
    ######### Replace with your codes #########
    # TODO: measure actual sizes of targets [width, depth, height] and update the dictionary of true target dimensions
    target_dimensions_dict = {'Orange': [.065,.065,.070], 'Lemon': [.075,.049,.049],
                              'Lime': [.070,.054,.054], 'tomato': [.069,.069,.060],
                              'Capsicum': [.070,.070,.090], 'Potato': [.090,.065,.060],
                              'Pumpkin': [.085,.085,.083], 'Garlic': [.065,.060,.0775]}
    #########

    # estimate target pose using bounding box and robot pose
    target_class = obj_info[0]     # get predicted target label of the box
    target_box = obj_info[1]       # get bounding box measures: [x,y,width,height]
    true_height = target_dimensions_dict[target_class][2]   # look up true height of by class label

    # compute pose of the target based on bounding box info, true object height, and robot's pose
    pixel_height = target_box[3]
    pixel_center = target_box[0]
    distance = true_height/pixel_height * focal_length  # estimated distance between the robot and the centre of the image plane based on height
    # image size 640x480 pixels, 640/2=320
    x_shift = 640/2 - pixel_center              # x distance between bounding box centre and centreline in camera view
    theta = np.arctan(x_shift/focal_length)     # angle of object relative to the robot
    ang = theta + robot_pose[2]     # angle of object in the world frame
    
   # relative object location
    distance_obj = distance/np.cos(theta) # relative distance between robot and object
    x_relative = distance_obj * np.cos(theta) # relative x pose
    y_relative = distance_obj * np.sin(theta) # relative y pose
    relative_pose = {'x': x_relative, 'y': y_relative}
    #print(f'relative_pose: {relative_pose}')

    # location of object in the world frame using rotation matrix
    delta_x_world = x_relative * np.cos(ang) - y_relative * np.sin(ang)
    delta_y_world = x_relative * np.sin(ang) + y_relative * np.cos(ang)
    # add robot pose with delta target pose
    target_pose = {'y': (robot_pose[1]+delta_y_world)[0],
                   'x': (robot_pose[0]+delta_x_world)[0]}
    #print(f'delta_x_world: {delta_x_world}, delta_y_world: {delta_y_world}')
    #print(f'target_pose: {target_pose}')
    

    return target_pose

def merge_estimations(target_pose_dict):
    """
    function:
        merge estimations of the same target
    input:
        target_pose_dict: dict, generated by estimate_pose
    output:
        target_est: dict, target pose estimations after merging
    """
    target_map = target_pose_dict
    print(target_map)
    capsicum_est, garlic_est, lemon_est, lime_est, orange_est, potato_est, pumpkin_est, tomato_est = [], [], [], [], [], [], [], []
    target_est = {}

    # combine the estimations from multiple detector outputs
    for key in target_map:
            #print(f)
            if key.startswith('Capsicum'):
                capsicum_est.append(
                    np.array(list(target_map[key].values()), dtype=float))
            elif key.startswith('Garlic'):
                garlic_est.append(
                    np.array(list(target_map[key].values()), dtype=float))
            elif key.startswith('Lemon'):
                lemon_est.append(
                    np.array(list(target_map[key].values()), dtype=float))
            elif key.startswith('Lime'):
                lime_est.append(
                    np.array(list(target_map[key].values()), dtype=float))
            elif key.startswith('Orange'):
                orange_est.append(
                    np.array(list(target_map[key].values()), dtype=float))
            elif key.startswith('Potato'):
                potato_est.append(
                    np.array(list(target_map[key].values()), dtype=float))
            elif key.startswith('Pumpkin'):
                pumpkin_est.append(
                    np.array(list(target_map[key].values()), dtype=float))
            elif key.startswith('tomato'):
                tomato_est.append(
                    np.array(list(target_map[key].values()), dtype=float))


    ######### Replace with your codes #########
    # TODO: the operation below takes the first three estimations of each target type, replace it with a better merge solution
    if len(capsicum_est) > 1:
        kmeans2 = KMeans(n_clusters=2, random_state=0).fit(capsicum_est)
        kmeans1 = KMeans(n_clusters=1, random_state=0).fit(capsicum_est)
        #print('apple ', kmeans1.inertia_ - kmeans2.inertia_)
        if kmeans1.inertia_ - kmeans2.inertia_ < 0.2:  # ! 1.5 is arbitrary, inertia is the sum of square distance
            capsicum_est = kmeans1.cluster_centers_
        else:
            capsicum_est = kmeans2.cluster_centers_

    if len(garlic_est) > 1:
        kmeans2 = KMeans(n_clusters=2, random_state=0).fit(garlic_est)
        kmeans1 = KMeans(n_clusters=1, random_state=0).fit(garlic_est)
        #print('lemon ', kmeans1.inertia_ - kmeans2.inertia_)
        if kmeans1.inertia_ - kmeans2.inertia_ < 0.2:  # ! 1.5 is arbitrary, inertia is the sum of square distance
            garlic_est = kmeans1.cluster_centers_
        else:
            garlic_est = kmeans2.cluster_centers_

    if len(lemon_est) > 1:
        kmeans2 = KMeans(n_clusters=2, random_state=0).fit(lemon_est)
        kmeans1 = KMeans(n_clusters=1, random_state=0).fit(lemon_est)
        #print('pear ', kmeans1.inertia_ - kmeans2.inertia_)
        if kmeans1.inertia_ - kmeans2.inertia_ < 0.2:  # ! 1.5 is arbitrary, inertia is the sum of square distance
            lemon_est = kmeans1.cluster_centers_
        else:
            lemon_est = kmeans2.cluster_centers_

    if len(lime_est) > 1:
        kmeans2 = KMeans(n_clusters=2, random_state=0).fit(lime_est)
        kmeans1 = KMeans(n_clusters=1, random_state=0).fit(lime_est)
        #print('orange ', kmeans1.inertia_ - kmeans2.inertia_)
        if kmeans1.inertia_ - kmeans2.inertia_ < 0.2:  # ! 1.5 is arbitrary, inertia is the sum of square distance
            lime_est = kmeans1.cluster_centers_
        else:
            lime_est = kmeans2.cluster_centers_

    if len(orange_est) > 1:
        kmeans2 = KMeans(n_clusters=2, random_state=0).fit(orange_est)
        kmeans1 = KMeans(n_clusters=1, random_state=0).fit(orange_est)
        #print('strawberry ', kmeans1.inertia_ - kmeans2.inertia_)
        if kmeans1.inertia_ - kmeans2.inertia_ < 0.2:  # ! 1.5 is arbitrary, inertia is the sum of square distance
            orange_est = kmeans1.cluster_centers_
        else:
            orange_est = kmeans2.cluster_centers_
    if len(potato_est) > 1:
        kmeans2 = KMeans(n_clusters=2, random_state=0).fit(potato_est)
        kmeans1 = KMeans(n_clusters=1, random_state=0).fit(potato_est)
        #print('pear ', kmeans1.inertia_ - kmeans2.inertia_)
        if kmeans1.inertia_ - kmeans2.inertia_ < 0.2:  # ! 1.5 is arbitrary, inertia is the sum of square distance
            potato_est = kmeans1.cluster_centers_
        else:
            potato_est = kmeans2.cluster_centers_

    if len(pumpkin_est) > 1:
        kmeans2 = KMeans(n_clusters=2, random_state=0).fit(pumpkin_est)
        kmeans1 = KMeans(n_clusters=1, random_state=0).fit(pumpkin_est)
        #print('orange ', kmeans1.inertia_ - kmeans2.inertia_)
        if kmeans1.inertia_ - kmeans2.inertia_ < 0.2:  # ! 1.5 is arbitrary, inertia is the sum of square distance
            pumpkin_est = kmeans1.cluster_centers_
        else:
            pumpkin_est = kmeans2.cluster_centers_

    if len(tomato_est) > 1:
        kmeans2 = KMeans(n_clusters=2, random_state=0).fit(tomato_est)
        kmeans1 = KMeans(n_clusters=1, random_state=0).fit(tomato_est)
        #print('strawberry ', kmeans1.inertia_ - kmeans2.inertia_)
        if kmeans1.inertia_ - kmeans2.inertia_ < 0.2:  # ! 1.5 is arbitrary, inertia is the sum of square distance
            tomato_est = kmeans1.cluster_centers_
        else:
            tomato_est = kmeans2.cluster_centers_

    for i in range(3):
        # except here is to deal with list with lenght of 1 ( out of indices problem)
        try:
            target_est['capsicum_' +
                       str(i)] = {'y': float(np.clip(capsicum_est[i][0],-1.5,1.5)), 'x': float(np.clip(capsicum_est[i][1],-1.5,1.5))}
        except:
            pass
        try:
            target_est['garlic_' +
                       str(i)] = {'y': float(np.clip(garlic_est[i][0],-1.5,1.5)), 'x': float(np.clip(garlic_est[i][1],-1.5,1.5))}
        except:
            pass
        try:
            target_est['lemon_' +
                       str(i)] = {'y': float(np.clip(lemon_est[i][0],-1.5,1.5)), 'x': float(np.clip(lemon_est[i][1],-1.5,1.5))}
        except:
            pass
        try:
            target_est['lime_' +
                       str(i)] = {'y': float(np.clip(lime_est[i][0],-1.5,1.5)), 'x': float(np.clip(lime_est[i][1],-1.5,1.5))}
        except:
            pass
        try:
            target_est['orange_' +
                       str(i)] = {'y': float(np.clip(orange_est[i][0],-1.5,1.5)), 'x': float(np.clip(orange_est[i][1],-1.5,1.5))}
        except:
            pass
        try:
            target_est['potato_' +
                       str(i)] = {'y': float(np.clip(potato_est[i][0],-1.5,1.5)), 'x': float(np.clip(potato_est[i][1],-1.5,1.5))}
        except:
            pass
        try:
            target_est['pumpkin_' +
                       str(i)] = {'y': float(np.clip(pumpkin_est[i][0],-1.5,1.5)), 'x': float(np.clip(pumpkin_est[i][1],-1.5,1.5))}
        except:
            pass
        try:
            target_est['tomato_' +
                       str(i)] = {'y': float(np.clip(tomato_est[i][0],-1.5,1.5)), 'x': float(np.clip(tomato_est[i][1],-1.5,1.5))}
        except:
            pass
    ###########################################

    return target_est

def parse_user_map(fname : str) -> dict:
    with open(fname, 'r') as f:
        usr_dict = ast.literal_eval(f.read())
        aruco_dict = {}
        for (i, tag) in enumerate(usr_dict["taglist"]):
            aruco_dict[tag] = np.reshape([usr_dict["map"][0][i],usr_dict["map"][1][i]], (2,1))
    return aruco_dict

def create_aruco_map(est):
    aruco_dict = {}
    for i in range(1, 11):
        #print(i)
        try:
            dict_number = {f'aruco{i}_0': {'x': float(np.squeeze(est[i][0])), 'y': float(np.squeeze(est[i][1]))}}
            aruco_dict = {**aruco_dict, **dict_number}
        except:
            pass
    return aruco_dict

# main loop
if __name__ == "__main__":
    run_number = str(input("Enter run number: "))
    script_dir = os.path.dirname(os.path.abspath(__file__))     # get current script directory (TargetPoseEst.py)

    # read in camera matrix
    fileK = f'{script_dir}/calibration/param/intrinsic.txt'
    camera_matrix = np.loadtxt(fileK, delimiter=',')

    # init YOLO model
    model_path = f'{script_dir}/YOLO/model/yolov8_model.pt'
    yolo = Detector(model_path)

    # create a dictionary of all the saved images with their corresponding robot pose
    image_poses = {}
    with open(f'{script_dir}/lab_output/images.txt') as fp:
        for line in fp.readlines():
            pose_dict = ast.literal_eval(line)
            image_poses[pose_dict['imgfname']] = pose_dict['pose']

    # estimate pose of targets in each image
    target_pose_dict = {}
    detected_type_list = []
    for image_path in image_poses.keys():
        input_image = cv2.imread(image_path)
        bounding_boxes, bbox_img = yolo.detect_single_image(input_image)
        # cv2.imshow('bbox', bbox_img)
        # cv2.waitKey(0)
        robot_pose = image_poses[image_path]

        for detection in bounding_boxes:
            # count the occurrence of each target type
            occurrence = detected_type_list.count(detection[0])
            target_pose_dict[f'{detection[0]}_{occurrence}'] = estimate_pose(camera_matrix, detection, robot_pose)

            detected_type_list.append(detection[0])

    # merge the estimations of the targets so that there are at most 3 estimations of each target type
    #print("merging")
    target_est = {}
    target_est = merge_estimations(target_pose_dict)
    target_est = {key.lower(): value for key, value in target_est.items()}
    print(target_est)
    # save target pose estimations
    #parser.add_argument('--slam-est', type=str, default='lab_output/slam.txt')
    aruco_est = parse_user_map('lab_output/slam.txt')
    aruco_map = create_aruco_map(aruco_est)
    with open(f'{script_dir}/lab_output/targets.txt', 'w') as fo:
        json.dump(target_est, fo, indent=4)
    with open(f'{script_dir}/Final_Maps_407/targets_run{run_number}_407.txt', 'w') as fo:
        json.dump(target_est, fo, indent=4)
    conc_map = {**aruco_map, **target_est}
    print('Estimations saved!')
    with open(f'{script_dir}/Map.txt', 'w') as fo:
        json.dump(conc_map, fo, indent=4)
    print('Peta terbuat!')
    with open(f'{script_dir}/lab_output/slam.txt', 'r') as f:
        usr_dict = ast.literal_eval(f.read())
    with open(f'{script_dir}/Final_Maps_407/slam_run{run_number}_407.txt', 'w') as fo:
        json.dump(usr_dict, fo, indent=1)