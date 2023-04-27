import os
import numpy as np
import re
import shutil
import pickle

def convert_boxes(txt_lines, angle_id):
    """
    Convert boxes from corners to [x,y,z,l,w,h,theta]
    """
    corners = []
    for i in range(8):
        corner = [float(co_str) for co_str in re.findall(r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", txt_lines[i])]
        corners.append(corner)
    corners = np.array(corners)
    location = (corners[0] + corners[7])*0.5
    dimension_x = np.linalg.norm(corners[0]-corners[4])
    dimension_y = np.linalg.norm(corners[0]-corners[3])
    dimension_z = np.linalg.norm(corners[0]-corners[1])
    dimension = np.array([dimension_x, dimension_y, dimension_z])
    if angle_id in range(10, 27):
        rotation_z = np.arctan((corners[0,1] - corners[4,1])/(corners[0,0] - corners[4,0])) + np.pi
        if rotation_z > np.pi:
            rotation_z = rotation_z - np.pi*2
    elif angle_id == 9:
        rotation_z = np.arctan((corners[0,1] - corners[4,1])/(corners[0,0] - corners[4,0]))
        if rotation_z > 0:
            rotation_z = rotation_z - np.pi
    elif angle_id == 27:
        rotation_z = np.arctan((corners[0,1] - corners[4,1])/(corners[0,0] - corners[4,0]))
        if rotation_z < 0:
            rotation_z = rotation_z + np.pi
    else: 
        rotation_z = np.arctan((corners[0,1] - corners[4,1])/(corners[0,0] - corners[4,0]))
    return location, dimension, rotation_z

def create_data_infos(pc_dir, lbl_dir, angle_test_set, num_angle, min_dist, max_dist, new_pc_dir, new_label_dir):
    """
    Create a list of infos for all samples in the simulation dataset.
    Give index to each point cloud sample (.bin) and save them into 'new_pc_dir'.
    Convert the label corresponding to the point cloud sample (.txt) and save them into 'new_label_dir'.
    """
    infos_train = []
    infos_val = []
    for i in range(num_angle):
        for j in range(min_dist, max_dist+1):
            angle_path = f"angle_{i}"
            dist_path = f"distance_{j}"
            this_label_path = os.path.join(lbl_dir, angle_path, dist_path, "label.txt")
            if not os.path.isfile(this_label_path):
                continue
            this_label_handle = open(this_label_path, "r")
            this_label_txt = this_label_handle.readlines()
            location, dimension, rotation_z = convert_boxes(this_label_txt, i)
            new_label = f"Car {location[0]} {location[1]} {location[2]} {dimension[0]} {dimension[1]} {dimension[2]} {rotation_z}"

            this_pc_dir = os.path.join(pc_dir, angle_path, dist_path)
            this_pc_all_files = os.listdir(this_pc_dir)

            for idx, pc_file in enumerate(this_pc_all_files):
                this_id = (i+36)*1000 + j*10 + idx
                new_pc_path = os.path.join(new_pc_dir, f'{this_id:06d}.bin')
                new_pc_path_relative = os.path.join("velodyne", f'{this_id:06d}.bin')
                shutil.copy(os.path.join(this_pc_dir, pc_file), new_pc_path)
                new_label_path = os.path.join(new_label_dir, f'{this_id:06d}.txt')
                with open(new_label_path, 'w') as f:
                    f.write(new_label)
                this_info = {}
                this_info['image'] = {}
                this_info['image']['image_idx'] = this_id
                this_info['point_cloud'] = {}
                this_info['point_cloud']['num_features'] = 4
                this_info['point_cloud']['velodyne_path'] = new_pc_path_relative
                this_info['annos'] = {}
                this_info['annos']['name'] = np.array(['Car'], dtype="U8")
                this_info['annos']['location'] = np.expand_dims(location, axis=0)
                this_info['annos']['dimensions'] = np.expand_dims(dimension, axis=0)
                this_info['annos']['rotation_z'] = np.array([rotation_z])
                if i in angle_test_set:
                    infos_val.append(this_info)
                else:
                    infos_train.append(this_info)
    return infos_train, infos_val

def main():
    # Please define all the parameters
    pc_dir = "data/simulation/L200"
    lbl_dir = "data/simulation/Clear_Simulations_labels_L200/"
    new_pc_dir = "data/kitti_format/training/velodyne"
    new_label_dir = "data/kitti_format/training/label_2"

    # angle_test_set = [8,17,26,35]
    angle_test_set = [6,15,24,33]
    num_angle = 36
    min_dist = 10
    max_dist = 96

    # Save the infos into these PKL files
    pkl_train = "data/kitti_format/simulated_train.pkl"
    pkl_val = "data/kitti_format/simulated_val.pkl"

    infos_train, infos_val = create_data_infos(pc_dir, lbl_dir, angle_test_set, num_angle, min_dist, max_dist, new_pc_dir, new_label_dir)

    with open(pkl_train, "wb") as f:
        pickle.dump(infos_train, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(pkl_val, "wb") as f:
        pickle.dump(infos_val, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()