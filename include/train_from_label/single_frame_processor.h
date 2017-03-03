#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/passthrough.h>
#include <pcl_ros/transforms.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/don.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/fpfh_omp.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


#include <cmath>

using namespace std;
using namespace cv;

float radius_l = 0.3;
float radius_s = 0.3;

struct feature_vector
{
    bool skip;
    float height_diff_l;
    float height_diff_s;
    pcl::Normal normal_l;
    pcl::Normal normal_s;

    pcl::FPFHSignature33 fpfh;
};

class Single_frame_processor
{
    public:
    Single_frame_processor();
    ~Single_frame_processor();

    vector<feature_vector> point_features;   // height diff, slope_1, slope_2

    pcl::PointCloud<pcl::PointXYZRGB> process_cloud    (pcl::PointCloud<pcl::PointXYZ> cloud, 
                                                        float map_width, float map_broad, float map_height, float map_resolution, float map_h_resolution, float robot_x, float robot_y);
    
    std::vector<int> get_neighbor_points(pcl::PointCloud<pcl::PointXYZ> cloud, float radius, int index);
    pcl::PointCloud<pcl::Normal>::Ptr calculateSurfaceNormal(pcl::PointCloud<pcl::PointXYZ>::Ptr input_point,
                                                        pcl::PointCloud<pcl::PointXYZ>::Ptr search_point,
                                                        float searchRadius );
    float compute_height_diff(pcl::PointCloud<pcl::PointXYZ> cloud, float radius, int index);
    vector<feature_vector>  compute_features(pcl::PointCloud<pcl::PointXYZ> cloud);

    pcl::PointCloud<pcl::PointXYZ> cloud_filter(pcl::PointCloud<pcl::PointXYZ> cloud);

    void reformCloud_cameraview(pcl::PointCloud<pcl::PointXYZ> cloud_base, 
                                                pcl::PointCloud<pcl::PointXYZ> cloud_camera, 
                                                Mat img_raw, Mat img_vision_label, string file_path);
                                        
};

Single_frame_processor::Single_frame_processor()
{
}

Single_frame_processor::~Single_frame_processor()
{

}


void Single_frame_processor::reformCloud_cameraview(pcl::PointCloud<pcl::PointXYZ> cloud_base, 
                                                pcl::PointCloud<pcl::PointXYZ> cloud_camera, 
                                                Mat img_raw, Mat img_vision_label, string file_path)
{
    cout << "in function" << endl;
    ofstream feature_file;
    string path = file_path + "_features.txt";
    cout << path << endl;
    feature_file.open (path.c_str());
    for(size_t i = 0; i < cloud_base.points.size(); i++)
    {
        pcl::PointXYZ point_cam = cloud_camera.points[i];
        if(point_cam.z < 0 )
            continue;

        // project points from camera fram to the image plane
        cv::Point3d pt_cv(point_cam.x, point_cam.y, point_cam.z);
        cv::Point2d uv = project3D_to_image(pt_cv);

        if(uv.x < 0 || uv.x >= 960 || uv.y < 0 || uv.y >= 540)
            continue; 

        int label = img_raw.at<uchar>(uv.y, uv.x);
        int label_vision = img_vision_label.at<uchar>(uv.y, uv.x);

        if(label == 0)
            continue;

        feature_file  << point_features[i].height_diff_l << " " << point_features[i].height_diff_s << " " ;  // height
        feature_file << point_features[i].normal_l.normal[0] << " " << point_features[i].normal_l.normal[1] << " " << point_features[i].normal_l.normal[2] << " ";
        feature_file << point_features[i].normal_s.normal[0] << " " << point_features[i].normal_s.normal[1] << " " << point_features[i].normal_s.normal[2] << " ";
        // for(size_t i= 0; i < 33; i++)
        // {
        //     feature_file << point_features[i].fpfh.histogram[i] << " ";
        // }
        feature_file  << uv.y << " " << uv.x <<"\n";
    }

    feature_file.close();

}

pcl::PointCloud<pcl::Normal>::Ptr Single_frame_processor::calculateSurfaceNormal(pcl::PointCloud<pcl::PointXYZ>::Ptr input_point,
                                                         pcl::PointCloud<pcl::PointXYZ>::Ptr search_point,
                                                         float searchRadius )
{
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud (input_point);
    ne.setSearchSurface(search_point);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
    ne.setSearchMethod (tree);

    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);

    ne.setRadiusSearch (searchRadius);
    ne.setViewPoint (0, 0, 1.5);
    ne.compute (*cloud_normals);

    return cloud_normals;
}

std::vector<int> Single_frame_processor::get_neighbor_points(pcl::PointCloud<pcl::PointXYZ> cloud, float radius, int index)
{
    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;

    pcl::PointCloud<pcl::PointXYZ>::Ptr          cloud_prt           (new pcl::PointCloud<pcl::PointXYZ>(cloud));
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;

    kdtree.setInputCloud (cloud_prt);
    kdtree.radiusSearch (cloud.points[index], radius, pointIdxRadiusSearch, pointRadiusSquaredDistance);

    return pointIdxRadiusSearch;
}

float Single_frame_processor::compute_height_diff(pcl::PointCloud<pcl::PointXYZ> cloud, float radius, int index)
{
    float max_h = -10;
    float min_h = 999;
    std::vector<int> pointIdxRadiusSearch = get_neighbor_points(cloud, radius, index);

    for (size_t i = 0; i < pointIdxRadiusSearch.size (); ++i)
    {
        pcl::PointXYZ point = cloud.points[ pointIdxRadiusSearch[i] ];
        if(point.z < min_h)
            min_h = point.z;
        if(point.z > max_h)
            max_h = point.z;
    }    

    return max_h - min_h;
}


pcl::PointCloud<pcl::PointXYZ> Single_frame_processor::cloud_filter(pcl::PointCloud<pcl::PointXYZ> cloud)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr  input_cloud       (new pcl::PointCloud<pcl::PointXYZ>(cloud));
    pcl::PointCloud<pcl::PointXYZ>::Ptr  cloud_passthrough (new pcl::PointCloud<pcl::PointXYZ>);

    cout << "before filter  " << input_cloud->points.size() << endl;

    pcl::PassThrough<pcl::PointXYZ> pass;
    // pass.setInputCloud (input_cloud);
    // pass.setFilterFieldName ("z");
    // pass.setFilterLimits (-map_height_/2, map_height_/2);
    // //pass.setFilterLimitsNegative (true);
    // pass.filter (*cloud_passthrough);
    // // cout << "after z filter  " << cloud_passthrough->points.size() << endl;

    pass.setInputCloud (input_cloud);
    pass.setFilterFieldName ("x");
    pass.setFilterLimits (-5.5, 5.5);
    pass.filter (*cloud_passthrough);
    cout << "after x filter  " << cloud_passthrough->points.size() << endl;

    pass.setInputCloud (cloud_passthrough);
    pass.setFilterFieldName ("y");
    pass.setFilterLimits (-5.5, 5.5);
    pass.filter (*cloud_passthrough);
    cout << "after y filter  " << cloud_passthrough->points.size() << endl;

    // pcl::VoxelGrid<pcl::PointXYZ> sor;
    // sor.setInputCloud (cloud_passthrough);
    // sor.setLeafSize (0.01, 0.01, 0.01);
    // sor.filter (*cloud_passthrough);
    // cout << "after voxel filter  " << cloud_passthrough->points.size() << endl;

    return *cloud_passthrough;
}

vector<feature_vector> Single_frame_processor::compute_features(pcl::PointCloud<pcl::PointXYZ> cloud)
{
    cloud = cloud_filter(cloud);
    pcl::PointCloud<pcl::Normal>::Ptr       normal_large        (new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::Normal>::Ptr       normal_small        (new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr          cloud_prt           (new pcl::PointCloud<pcl::PointXYZ>(cloud));
    // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ground_prt    (new pcl::PointCloud<pcl::PointXYZ>(ground_points_));



    cout << "normal computation....... ";
    // normal_large = calculateSurfaceNormal(cloud_prt, cloud_prt, radius_l);
    normal_small = calculateSurfaceNormal(cloud_prt, cloud_prt, radius_s);
    cout << "done" << endl;
    ///////////////////////////////////////////////////////////////////////////////////////////
    // Create the FPFH estimation class, and pass the input dataset+normals to it
    pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
    fpfh.setInputCloud (cloud_prt);
    fpfh.setInputNormals (normal_small);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    fpfh.setSearchMethod (tree);
    // Output datasets
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs (new pcl::PointCloud<pcl::FPFHSignature33> ());
    // Use all neighbors in a sphere of radius 5cm
    // IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
    fpfh.setRadiusSearch (radius_l);
    fpfh.compute (*fpfhs);


    for (size_t i = 0; i < cloud.points.size(); i = i+1)
    {
        feature_vector point_feature;
        // cout << cloud.points[i].x << endl;
        if(abs(cloud.points[i].x)>5 ||  abs(cloud.points[i].y)>5 )
        {

            point_feature.skip = true;
            point_features.push_back(point_feature);
            continue;
        }    
        point_feature.height_diff_l = compute_height_diff(cloud, radius_l, i);
        point_feature.height_diff_s = compute_height_diff(cloud, radius_s, i);
        // point_feature.normal_l = normal_large->points[i];
        point_feature.normal_s = normal_small->points[i];
        // point_feature.fpfh = fpfhs->points[i];

        point_features.push_back(point_feature);
    }
        cout << "done" << endl;
    return point_features;
}

pcl::PointCloud<pcl::PointXYZRGB> Single_frame_processor::process_cloud(pcl::PointCloud<pcl::PointXYZ> cloud, 
                        float map_width, float map_broad, float map_height, float map_resolution, float map_h_resolution, float robot_x, float robot_y)
{
    cout << "in process cloud   " << endl;
    point_features.clear();
    point_features = compute_features(cloud);

    pcl::PointCloud<pcl::PointXYZRGB> cloud_color;
    copyPointCloud(cloud, cloud_color);
    return cloud_color;
}


