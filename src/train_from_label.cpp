#include "ros/ros.h"
#include <tf/transform_listener.h>
#include <sensor_msgs/point_cloud_conversion.h>

#include <cv_bridge/cv_bridge.h>
#include <sstream>
#include <train_from_label/terrain_feature_estimator.h>


tf::TransformListener* tfListener = NULL;
Cloud_Matrix_Loador* cml;

ros::Publisher  pub_cloud;
bool initialized = false;

pcl::PointCloud<pcl::PointXYZRGB> ground_cloud1_;

float robot_x_, robot_y_;

string map_frame = "world_corrected";
string output_frame = "base_link_oriented";
string process_frame = "base_link";
// string process_frame = "world_corrected";

Mat output_label, true_label;
string feature_img_path;
bool img_ready = false;
double label_time = 0;

vector<string> files_label;
vector<string> files_label_vision;
vector<string> files_feature;
vector<cv::Mat> h_labeled_imgs;
int file_index = 2;

void publish(ros::Publisher pub, pcl::PointCloud<pcl::PointXYZ> cloud, int type = 2)
{
    sensor_msgs::PointCloud2 pointlcoud2;
    pcl::toROSMsg(cloud, pointlcoud2);

    if(type == 2)
    { 
        pub.publish(pointlcoud2);
    }
    else
    {
        sensor_msgs::PointCloud pointlcoud;
        sensor_msgs::convertPointCloud2ToPointCloud(pointlcoud2, pointlcoud);

        pointlcoud.header = pointlcoud2.header;
        pub.publish(pointlcoud);
    }

}

void publish(ros::Publisher pub, pcl::PointCloud<pcl::PointXYZRGB> cloud, int type = 2)
{
    sensor_msgs::PointCloud2 pointlcoud2;
    pcl::toROSMsg(cloud, pointlcoud2);

    pub.publish(pointlcoud2);
}

sensor_msgs::PointCloud2 transform_cloud(sensor_msgs::PointCloud2 cloud_in, string frame_target)
{
    ////////////////////////////////// transform ////////////////////////////////////////
    sensor_msgs::PointCloud2 cloud_out;
    tf::StampedTransform to_target;

    try 
    {
        // tf_listener_->waitForTransform(frame_target, cloud_in.header.frame_id, cloud_in.header.stamp, ros::Duration(1.0));
        // tf_listener_->lookupTransform(frame_target, cloud_in.header.frame_id, cloud_in.header.stamp, to_target);
        tfListener->lookupTransform(frame_target, cloud_in.header.frame_id, cloud_in.header.stamp, to_target);
    }
    catch (tf::TransformException& ex) 
    {
        ROS_WARN("[draw_frames] TF exception:\n%s", ex.what());
        // return cloud_in;
    }

    Eigen::Matrix4f eigen_transform;
    pcl_ros::transformAsMatrix (to_target, eigen_transform);
    pcl_ros::transformPointCloud (eigen_transform, cloud_in, cloud_out);

    cloud_out.header.frame_id = frame_target;
    return cloud_out;
}


void callback_cloud(const sensor_msgs::PointCloud2ConstPtr &cloud_in)
{
    if(!img_ready)
        return; 

    cout << cloud_in->header.stamp;
    cout << "cloud recieved: " << ros::Time::now() << endl;
    sensor_msgs::PointCloud2 cloud_transformed = transform_cloud(*cloud_in, process_frame);
    sensor_msgs::PointCloud2 cloud_camera = transform_cloud(*cloud_in, "kinect2_rgb_optical_frame");

    if(cloud_transformed.height == 0 && cloud_transformed.width == 0)
        return;

    pcl::PointCloud<pcl::PointXYZ> pcl_cloud, pcl_cloud_camera;
    pcl::fromROSMsg(cloud_transformed, pcl_cloud);
    pcl::fromROSMsg(cloud_camera, pcl_cloud_camera);

    ros::Time begin = ros::Time::now();

    ground_cloud1_ = cml->process_cloud(pcl_cloud, 16, 16, 6, 0.15, 0.015, 0, 0);
    ground_cloud1_.header.frame_id = process_frame;

    // get result in camera img
    Mat label_vision = cv::imread(files_label_vision[file_index], CV_LOAD_IMAGE_GRAYSCALE);
    cout << files_label_vision[file_index] << endl;
    cv::resize(label_vision, label_vision, cv::Size(960, 540));
    cv::imshow("true_label", label_vision);
    cv::waitKey(10); 

    Mat label_camera = cml->reformCloud_cameraview(pcl_cloud, pcl_cloud_camera, cml->cost_map_, true_label, label_vision, files_feature[file_index]);
    cv::imshow("result", label_camera);
    cv::waitKey(10);

    cout << ros::Time::now() - begin << "  loaded cloud *********************" << endl;

    publish(pub_cloud, ground_cloud1_); // publishing colored points with defalt cost function  
    img_ready = false;
}

void image_callback(sensor_msgs::ImageConstPtr image_msg)
{
    // double diff = image_msg->header.stamp.toSec() - label_time;
    // cout << image_msg->header.stamp << " " << diff << endl;

    // if(diff == 0)
    // {
    //     output_label = cv_bridge::toCvShare(image_msg, "bgr8")->image;
    //     img_ready = true;
    // }

    // cout << "in" << endl;
    Mat new_img = cv_bridge::toCvShare(image_msg, "bgr8")->image;

    // string file_path_color = files_color[file_index];
    string file_path_label = files_label[file_index];

    // cout << file_path_color << " " << file_path_label << endl;

    // Mat true_color         = cv::imread(file_path_color);
    true_label             = cv::imread(file_path_label);

    // cv::imshow("matched_label", new_img);
    // cv::imshow("true_label", true_label);
    // // cout << file_path_color << endl;
    // cv::waitKey(10); 

    Mat gray_newimg;
    cv::cvtColor(new_img, gray_newimg, cv::COLOR_BGR2GRAY);
    // cv::cvtColor(true_color, gray_label, cv::COLOR_BGR2GRAY);

    for(int i = 1; i< h_labeled_imgs.size(); i++)
    {
        Mat diff = abs(gray_newimg - h_labeled_imgs[i]);
        int diff_pix = countNonZero(diff);
        // cout << diff_pix << endl;
        // cv::imshow("matched_label", gray_newimg);
        // cv::imshow("true_label", h_labeled_imgs[i]);
        // // cout << file_path_color << endl;
        // cv::waitKey(10); 

        if(diff_pix < 420000)
        {
            cout << image_msg->header.stamp << endl;
            cout << file_path_label << endl;

            cv::imshow("matched_label", gray_newimg);
            cv::imshow("true_label", h_labeled_imgs[i]);
            // cout << file_path_color << endl;
            cv::waitKey(10); 

            img_ready = true;
            file_index = i;
        }
    }
    
    // cv::imshow("file_path_color", true_color);
    // cv::waitKey(10);    
    // cout << "out" << endl;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "geometric_classifier");
    ros::NodeHandle node; 
    ros::NodeHandle private_nh("~");
    
    private_nh.getParam("stamp", label_time);
    // private_nh.getParam("label_path", label_path);

    // true_label = cv::imread(label_path);
    // cv::imshow("ground_truth", true_label);
    // cv::waitKey(10);

    double ti = ros::Time::now().toSec();
    cout << label_time << " " << ti << " " << ros::Time::now() <<endl;
    cout << label_time - ti << endl;
    cml = new Cloud_Matrix_Loador();

    tfListener = new (tf::TransformListener);

    ros::Subscriber sub_cloud  = node.subscribe<sensor_msgs::PointCloud2>("/points_raw", 1, callback_cloud);
    ros::Subscriber sub_image  = node.subscribe<sensor_msgs::Image>("/kinect2/qhd/image_color", 1, image_callback);

    // ros::Subscriber sub_velodyne_left  = node.subscribe<sensor_msgs::PointCloud2>("/ndt_map", 1, callback_cloud);
    pub_cloud      = node.advertise<sensor_msgs::PointCloud2>("/cloud_filtered", 1);

    string rosbag_name = "snow_grass";
    for(int i = 0; i< 12; i++)
    {
        string name_label = "/home/xi/workspace/labels/";
        string name_feature = "/home/xi/workspace/labels/output/";

        name_label = name_label + rosbag_name + "/" + rosbag_name + "_";
        name_feature = name_feature + rosbag_name + "/" + rosbag_name + "_";

        std::stringstream ss;
        ss << i;
        std::string str;
        ss >> str;

        name_label = name_label + str;
        string color_path = name_label + "_image.png";
        string label_path = name_label + "_label.png";
        files_label.push_back(label_path);

        string vision_path = name_label;
        files_feature.push_back(name_feature + str);
        files_label_vision.push_back(vision_path + "_vision.png");

        cout << color_path <<  " " << vision_path + "_vision.png" << " " << name_feature + str << endl;
        Mat true_color = cv::imread(color_path, CV_LOAD_IMAGE_GRAYSCALE);
        h_labeled_imgs.push_back(true_color);
        //         cv::imshow("true_label", true_color);
        // // cout << file_path_color << endl;
        // cv::waitKey(10); 
    }


    ros::spin();

    return 0;
}
