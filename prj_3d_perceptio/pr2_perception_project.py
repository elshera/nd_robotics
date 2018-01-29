#!/usr/bin/env python

#---------------------------------------
# Import Required Modules
#---------------------------------------
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *
import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml
import os


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict                   = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]       = arm_name.data
    yaml_dict["object_name"]    = object_name.data
    yaml_dict["pick_pose"]      = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"]     = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# prefiltering operation of the cloud
def pcl_filter(pcl_msg):
    # ROS callback function for the Point Cloud Subscriber. Takes a point cloud and
    # performs object recognition on it. 
    rospy.loginfo("Received new cloud data!\n")

    #---------------------------------------
    # TODO: Convert ROS msg to PCL data
    #---------------------------------------
    cloud = ros_to_pcl(pcl_msg)
    # save the original cloud
    pcl.save(cloud, 'cloud_original.pcd')

    #---------------------------------------
    # TODO: Statistical Outlier Filtering. We need to remove as much noise as possible so it make sense to apply a filtering before.
    #---------------------------------------
    outlier_filter = cloud.make_statistical_outlier_filter()
    # Set the number of neighboring points to analyze for any given point
    outlier_filter.set_mean_k(50)
    # Any point with a mean distance larger than global will be considered outlier
    outlier_filter.set_std_dev_mul_thresh(1.0)
    cloud_filtered = outlier_filter.filter()
    # save filter for checking/debug purpose
    pcl.save(cloud_filtered, 'cloud_filtered.pcd')

    #---------------------------------------
    # TODO: Voxel Grid Downsampling, reduce computation of the scene
    #---------------------------------------
    vox = cloud_filtered.make_voxel_grid_filter()
    # voxel size
    LEAF_SIZE = 0.005  #  5 mm seems to be a good choice to not miss features
    # Set the voxel size in the filter object
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    # Call the filter function to obtain the resultant downsampled point cloud
    cloud_voxeled = vox.filter()
    # save filter for checking/debug purpose
    pcl.save(cloud_voxeled, 'cloud_voxeled.pcd')

    #---------------------------------------
    # TODO: PassThrough Filter
    #---------------------------------------
    # create filter object, on z axis
    passthrough_z = cloud_voxeled.make_passthrough_filter()
    passthrough_z.set_filter_field_name('z')
    passthrough_z.set_filter_limits(0.6, 1.1)
    cloud_passthrough_z = passthrough_z.filter()

    # create filter object, on y axis
    passthrough_y = cloud_passthrough_z.make_passthrough_filter()
    passthrough_y.set_filter_field_name('y')
    passthrough_y.set_filter_limits(-0.5, 0.5)
    cloud_passthrough_y = passthrough_y.filter()

    # create filter object, on x axis
    passthrough_x = cloud_passthrough_y.make_passthrough_filter()
    passthrough_x.set_filter_field_name('x')
    passthrough_x.set_filter_limits(0.34, 1.0)
    cloud_passthrough = passthrough_x.filter()
    # save filter for checking/debug purpose
    pcl.save(cloud_passthrough, 'cloud_passthrough.pcd')
    return cloud_passthrough

# segmentation & clustering
def segmentation(pcl_msg):
    cloud_passthrough = pcl_filter(pcl_msg)

    #---------------------------------------
    # TODO: RANSAC Plane Segmentation
    #---------------------------------------
    # create segmentation object
    seg = cloud_passthrough.make_segmenter()
    # Set the model you wish to fit 
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    # Max distance for a point to be considered fitting the model Experiment with different values for max_distance for segmenting the table
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)
    # Call the segment function to obtain set of inlier indices and model coefficients
    inliers, coefficients = seg.segment()

    #---------------------------------------
    # TODO: Extract inliers and outliers
    #---------------------------------------
    table   = cloud_passthrough.extract(inliers, negative=False)
    objects = cloud_passthrough.extract(inliers, negative=True)
    # save filter for checking/debug purpose
    pcl.save(table, 'cloud_table.pcd')
    pcl.save(objects, 'cloud_objects.pcd')

    #---------------------------------------
    # TODO: Euclidean Clustering, this is needed to separate the objects, this ends the segmentation process
    #---------------------------------------
    white_cloud = XYZRGB_to_XYZ(objects)
    tree = white_cloud.make_kdtree()
    # Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
    # Set tolerances for distance threshold as well as minimum and maximum cluster size (in points)
    ec.set_ClusterTolerance(0.01)
    ec.set_MinClusterSize(100)
    ec.set_MaxClusterSize(5000)
    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()
    # print number of clusters
    print "number of clusters:",len(cluster_indices)

    #---------------------------------------
    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
    #---------------------------------------
    # Assign a random color to each isolated object in the scene
    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []
    for j, indices in enumerate(cluster_indices):
        for i, idx in enumerate(indices):    
            color_cluster_point_list.append([white_cloud[idx][0],
                                             white_cloud[idx][1],             
                                             white_cloud[idx][2], 
                                             rgb_to_float(cluster_color[j])])


    # Create new cloud containing all clusters, each with unique color
    pcl_cloud_colored = pcl.PointCloud_PointXYZRGB()
    pcl_cloud_colored.from_list(color_cluster_point_list)
    pcl.save(pcl_cloud_colored, 'pcl_cloud_colored.pcd')

    #---------------------------------------
    # TODO: Convert new cliud to ROS messages
    #---------------------------------------
    ros_objects_coloured   = pcl_to_ros(pcl_cloud_colored)
    ros_objects            = pcl_to_ros(objects)
    ros_table              = pcl_to_ros(table)

    #---------------------------------------
    # TODO: Publish ROS messages
    #---------------------------------------
    pcl_colored_objects_pub.publish(ros_objects_coloured)        # solid colored objects
    pcl_objects_pub.publish(ros_objects)                         # original color objects
    pcl_table_pub.publish(ros_table)                             # table cloud

    return objects, cluster_indices, white_cloud

# classification and detection function
def classification(pcl_msg):
    objects, cluster_indices, white_cloud = segmentation(pcl_msg)  

    #---------------------------------------
    # Classify the clusters! (loop through each detected cluster one at a time)
    #---------------------------------------
    # Store the detected objects and labels in thesevc  lists
    detected_objects_labels  = []
    detected_objects_list    = []

    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster from the extracted outliers (cloud_objects)
        pcl_cluster = objects.extract(pts_list) # <type 'pcl._pcl.PointCloud_PointXYZRGB'>
        # TODO: convert the cluster from pcl to ROS using helper function
        ros_cluster = pcl_to_ros(pcl_cluster)
        # Extract histogram features
        histogram_bins = 128
        chists  = compute_color_histograms(ros_cluster, nbins=histogram_bins, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists  = compute_normal_histograms(normals, nbins=histogram_bins)
        feature = np.concatenate((chists, nhists))

        # Make the prediction, retrieve the label for the result and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        pcl_object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects_list.append(do)

    # Publish the list of detected objects
    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
    return detected_objects_list, detected_objects_labels

# callback function for the subscriber
def pcl_callback(pcl_msg):
    #---------------------------------------
    # get objects and collision points
    #---------------------------------------
    detected_objects_list, detected_objects_labels = classification(pcl_msg)

    #---------------------------------------
    # Publish the list of detected objects & call pr2_mover funtion
    #---------------------------------------    
    if detected_objects_list:
        pcl_detected_objects_pub.publish(detected_objects_list)
        object_list_param  = rospy.get_param('/object_list')
        # Check consistency of detected objects list
        if not len(detected_objects_list) == len(object_list_param):
            rospy.loginfo("List of detected objects does not match pick list.")
            return
        else:
            try:
                pr2_mover(detected_objects_list)
            except rospy.ROSInterruptException:
                pass
    else:
        rospy.loginfo("No objects detected.")

# function to load parameters and request PickPlace service
def pr2_mover(detected_objects_list):
    #---------------------------------------
    # TODO: Get/Read parameters
    #---------------------------------------
    object_list_param  = rospy.get_param('/object_list')

    # get additional paramters
    num_objects         = len(object_list_param)
    num_scene           = 2 #rospy.get_param('/test_scene_num')
    test_scene_num      = Int32()
    test_scene_num.data = num_scene

    # Initialize dropbox positions from ROS parameter
    dropbox_list_param     = rospy.get_param('/dropbox')
    red_dropbox_position   = dropbox_list_param[0]['position']
    green_dropbox_position = dropbox_list_param[1]['position']

    # evaluate accuracy of the prediction
    hit_count = 0
    # Create list of ground truth labels
    true_labels = [element['name'] for element in object_list_param]

    # For each detected object, compare the predicted label with the ground truth from the pick list.
    for detected_object in detected_objects_list:
        predicted_label = detected_object.label
        if predicted_label in true_labels:
            true_labels.remove(predicted_label)
            hit_count += 1
        else:
            detected_object.label = 'error'
    rospy.loginfo('Detected {} objects out of {}.'.format(hit_count, num_objects))


    # Create list of detected objects sorted in the order of the pick list
    sorted_objects = []
    # Iterate over the pick list
    for i in range(num_objects):
        # take the label of the pick list item
        item_label = object_list_param[i]['name']
        # Find detected object corresponding to pick list item
        for detected_object in detected_objects_list:
            if detected_object.label == item_label:
                 # Append detected object to sorted_objects list
                sorted_objects.append(detected_object)
                # Remove current object
                detected_objects_list.remove(detected_object)
                break


    # Create lists for centroids and dropbox groups 
    centroids      = []
    dropbox_groups = []
    for sorted_object in sorted_objects:
        # Calculate the centroid
        pts = ros_to_pcl(sorted_object.cloud).to_array()
        centroid = np.mean(pts, axis=0)[:3]
        # Append centroid as <numpy.float64> data type
        centroids.append(centroid)
        # Search for the matching dropbox group, assuming 1:1 correspondence between sorted objects and pick list
        for pl_item in object_list_param:
            # Compare objects by their label
            if pl_item['name'] == sorted_object.label:
                # Matching object found, add the group to the list
                dropbox_groups.append(pl_item['group'])
                break

    # Initialize list of request parameters for later output to yaml format
    request_params = []

    #---------------------------------------
    # TODO: Loop through the pick list
    #---------------------------------------
    # Iterate over detected objects to generate ROS message for each object
    for j in range(len(sorted_objects)):
        # Create 'object_name' message with label as native string type
        object_name = String()
        object_name.data = str(sorted_objects[j].label)

        # Initialize the dropbox group
        object_group = dropbox_groups[j]

        # Create 'arm_name' message 
        arm_name = String()

        # Select right arm for green group and left arm for red group
        arm_name.data = 'right' if object_group == 'green' else 'left'

        # Convert <numpy.float64> data type to native float as expected by ROS
        np_centroid = centroids[j]
        scalar_centroid = [np.asscalar(element) for element in np_centroid]

        # Create 'pick_pose' message with centroid as the position data
        pick_pose = Pose()
        pick_pose.position.x = scalar_centroid[0]
        pick_pose.position.y = scalar_centroid[1]
        pick_pose.position.z = scalar_centroid[2]

        # Create 'place_pose' message with dropbox center as position data
        place_pose = Pose()
        dropbox_position = green_dropbox_position if object_group == 'green' else red_dropbox_position
        place_pose.position.x = dropbox_position[0]
        place_pose.position.y = dropbox_position[1]
        place_pose.position.z = dropbox_position[2]

        # Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
        request_params.append(make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose))

        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')

        # Call 'pick_place_routine' service
        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)
            response = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)
            print("Response: {}".format(response.success))
        except rospy.ServiceException, e:
            print("Service call failed: {}".format(e))

    #---------------------------------------
    # TODO: Output your request parameters into output yaml file
    #---------------------------------------
    file_name = "output_{}.yaml".format(num_scene)
    if not os.path.exists(file_name):
        send_to_yaml(file_name, request_params)
        print(file_name + " saved!")

if __name__ == '__main__':
    #---------------------------------------
    # TODO: ROS node initialization
    #---------------------------------------
    rospy.init_node('perception_project')

    #---------------------------------------
    # TODO: Create Subscribers
    #---------------------------------------
    subs = rospy.Subscriber('/pr2/world/points', pc2.PointCloud2, pcl_callback, queue_size=1)

    #---------------------------------------
    # TODO: Create Publishers
    #---------------------------------------
    # Isolated object point cloud with the object's original colors
    pcl_objects_pub         = rospy.Publisher('/pcl_objects', PointCloud2, queue_size=1)
    # Isolated object point cloud with random colors
    pcl_colored_objects_pub = rospy.Publisher('/pcl_world', PointCloud2, queue_size=1)
    # Table point cloud without the objects
    pcl_table_pub           = rospy.Publisher('/pcl_table', PointCloud2, queue_size=1)
    # detected object 
    pcl_detected_objects_pub= rospy.Publisher('/detected_objects', DetectedObjectsArray, queue_size=1)
    # object markers
    pcl_object_markers_pub  = rospy.Publisher('/object_markers', Marker,queue_size=1)

    #---------------------------------------
    # TODO: Load Model From disk
    #---------------------------------------
    model            = pickle.load(open('model.sav', 'rb'))
    clf              = model['classifier']
    encoder          = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler           = model['scaler']

    #---------------------------------------
    # Initialize color_list
    #---------------------------------------
    get_color_list.color_list = []

    #---------------------------------------
    # TODO: Spin while node is not shutdown
    #---------------------------------------
    while not rospy.is_shutdown():
        rospy.spin()
