//=============================================================================
//
// akaze_inliers.cpp
// Authors: Pablo F. Alcantarilla, Jesus Nuevo, Nikolay Zakharov
// Institutions: TrueVision Solutions (2)
// Date: 07/10/2014
// Email: pablofdezalc@gmail.com
//
// AKAZE Features Copyright 2014 Pablo F. Alcantarilla, Jesus Nuevo; 2016 Nikolay Zakharov
// All Rights Reserved
// See LICENSE for the license information
//=============================================================================

/**
 * @file akaze_inliers.cpp
 * @brief Main program for matching two images with AKAZE features
 * @date Oct 07, 2014
 * @author Pablo F. Alcantarilla
 */

#include "./lib/AKAZE.h"

// OpenCV
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <fstream>

using namespace std;

/* ************************************************************************* */
// Image matching options
const float MIN_H_ERROR = 2.50f;            ///< Maximum error in pixels to accept an inlier
const float DRATIO = 0.80f;                 ///< NNDR Matching value

/* ************************************************************************* */
/**
 * @brief This function parses the command line arguments for setting A-KAZE parameters
 * and image matching between two input images
 * @param options Structure that contains A-KAZE settings
 * @param img_path1 Path for the first input image
 * @param img_path2 Path for the second input image
 * @param inliers_path Path for the file to save inliers
 */
int parse_input_options(AKAZEOptions &options, std::string &img_path1,
                        std::string &img_path2, std::string &homography_path,
                        int argc, char *argv[]);

int save_inliers(const string& outFile, const vector<cv::Point2f> ptpairs);

/* ************************************************************************* */
int main(int argc, char *argv[]) {

    // Variables
    AKAZEOptions options;
    cv::Mat img1, img1_32, img2, img2_32, img1_rgb, img2_rgb, img_com, img_r;
    string img_path1, img_path2, inliers_path;
    float rfactor = .60;

    vector<cv::KeyPoint> kpts1, kpts2;
    vector<vector<cv::DMatch> > dmatches;
    cv::Mat desc1, desc2;
    cv::Mat HG;

    // Parse the input command line options
    if (parse_input_options(options, img_path1, img_path2, inliers_path, argc, argv))
        return -1;

    // Read image 1 and if necessary convert to grayscale.
    img1 = cv::imread(img_path1, 0);
    if (img1.data == NULL) {
        cerr << "Error loading image 1: " << img_path1 << endl;
        return -1;
    }

    // Read image 2 and if necessary convert to grayscale.
    img2 = cv::imread(img_path2, 0);
    if (img2.data == NULL) {
        cerr << "Error loading image 2: " << img_path2 << endl;
        return -1;
    }

    // Convert the images to float
    img1.convertTo(img1_32, CV_32F, 1.0 / 255.0, 0);
    img2.convertTo(img2_32, CV_32F, 1.0 / 255.0, 0);

    // Color images for results visualization
    img1_rgb = cv::Mat(cv::Size(img1.cols, img1.rows), CV_8UC3);
    img2_rgb = cv::Mat(cv::Size(img2.cols, img1.rows), CV_8UC3);
    img_com = cv::Mat(cv::Size(img1.cols * 2, img1.rows), CV_8UC3);
    img_r = cv::Mat(cv::Size(img_com.cols * rfactor, img_com.rows * rfactor), CV_8UC3);

    // Create the first AKAZE object
    options.img_width = img1.cols;
    options.img_height = img1.rows;
    libAKAZE::AKAZE evolution1(options);

    // Create the second HKAZE object
    options.img_width = img2.cols;
    options.img_height = img2.rows;
    libAKAZE::AKAZE evolution2(options);

    evolution1.Create_Nonlinear_Scale_Space(img1_32);
    evolution1.Feature_Detection(kpts1);
    evolution1.Compute_Descriptors(kpts1, desc1);

    evolution2.Create_Nonlinear_Scale_Space(img2_32);
    evolution2.Feature_Detection(kpts2);
    evolution2.Compute_Descriptors(kpts2, desc2);

    // Matching Descriptors!!
    vector<cv::Point2f> matches, inliers;
    cv::Ptr<cv::DescriptorMatcher> matcher_l2 = cv::DescriptorMatcher::create("BruteForce");
    cv::Ptr<cv::DescriptorMatcher> matcher_l1 = cv::DescriptorMatcher::create("BruteForce-Hamming");

    if (options.descriptor < MLDB_UPRIGHT)
        matcher_l2->knnMatch(desc1, desc2, dmatches, 2);
    else
        matcher_l1->knnMatch(desc1, desc2, dmatches, 2);

    // Compute Inliers!!
    matches2points_nndr(kpts1, kpts2, dmatches, matches, DRATIO);

    compute_inliers_ransac(matches, inliers, MIN_H_ERROR, false);

    // Save inliers!!
    save_inliers(inliers_path, inliers);
}

/* ************************************************************************* */
int save_inliers(const string& outFile, const vector<cv::Point2f> ptpairs) {
    int x1, y1;
    int x2, y2;

    ofstream ipfile(outFile.c_str());

    if (!ipfile) {
        cerr << "Couldn't open file '" << outFile << "'!" << endl;
        return -1;
    }

    ipfile << "{\"points\": [" << endl;

    for (size_t i = 0; i < ptpairs.size(); i+= 2) {
        x1 = (int)(ptpairs[i].x+.5);
        y1 = (int)(ptpairs[i].y+.5);
        x2 = (int)(ptpairs[i+1].x+.5);
        y2 = (int)(ptpairs[i+1].y+.5);

        if (i > 0)
            ipfile << ", ";

        ipfile << "{"
               << "\"pattern_point\": {\"x\": " << x1 << ", \"y\": " << y1 << "}, "
               << "\"image_point\": {\"x\": " << x2 << ", \"y\": " << y2 << "}"
               << "}"
               << endl;
    }

    ipfile << "]}" << endl;

    // Close the txt file
    ipfile.close();

    return 0;
}

/* ************************************************************************* */
int parse_input_options(AKAZEOptions &options, std::string &img_path1, std::string &img_path2,
                        std::string& inliers_path, int argc, char *argv[]) {

    // If there is only one argument return
    if (argc == 1) {
        show_input_options_help(1);
        return -1;
    }
        // Set the options from the command line
    else if (argc >= 2) {
        inliers_path = "./inliers.json";

        // Load the default options
        options = AKAZEOptions();

        if (!strcmp(argv[1], "--help")) {
            show_input_options_help(1);
            return -1;
        }

        img_path1 = argv[1];
        img_path2 = argv[2];

        for (int i = 3; i < argc; i++) {
            if (!strcmp(argv[i], "--soffset")) {
                i = i + 1;
                if (i >= argc) {
                    cerr << "Error introducing input options!!" << endl;
                    return -1;
                } else {
                    options.soffset = atof(argv[i]);
                }
            } else if (!strcmp(argv[i], "--omax")) {
                i = i + 1;
                if (i >= argc) {
                    cerr << "Error introducing input options!!" << endl;
                    return -1;
                } else {
                    options.omax = atof(argv[i]);
                }
            } else if (!strcmp(argv[i], "--dthreshold")) {
                i = i + 1;
                if (i >= argc) {
                    cerr << "Error introducing input options!!" << endl;
                    return -1;
                } else {
                    options.dthreshold = atof(argv[i]);
                }
            } else if (!strcmp(argv[i], "--sderivatives")) {
                i = i + 1;
                if (i >= argc) {
                    cerr << "Error introducing input options!!" << endl;
                    return -1;
                } else {
                    options.sderivatives = atof(argv[i]);
                }
            } else if (!strcmp(argv[i], "--nsublevels")) {
                i = i + 1;
                if (i >= argc) {
                    cerr << "Error introducing input options!!" << endl;
                    return -1;
                } else {
                    options.nsublevels = atoi(argv[i]);
                }
            } else if (!strcmp(argv[i], "--diffusivity")) {
                i = i + 1;
                if (i >= argc) {
                    cerr << "Error introducing input options!!" << endl;
                    return -1;
                } else {
                    options.diffusivity = DIFFUSIVITY_TYPE(atoi(argv[i]));
                }
            } else if (!strcmp(argv[i], "--descriptor")) {
                i = i + 1;
                if (i >= argc) {
                    cerr << "Error introducing input options!!" << endl;
                    return -1;
                } else {
                    options.descriptor = DESCRIPTOR_TYPE(atoi(argv[i]));

                    if (options.descriptor < 0 || options.descriptor > MLDB) {
                        options.descriptor = MLDB;
                    }
                }
            } else if (!strcmp(argv[i], "--descriptor_channels")) {
                i = i + 1;
                if (i >= argc) {
                    cerr << "Error introducing input options!!" << endl;
                    return -1;
                } else {
                    options.descriptor_channels = atoi(argv[i]);

                    if (options.descriptor_channels <= 0 || options.descriptor_channels > 3) {
                        options.descriptor_channels = 3;
                    }
                }
            } else if (!strcmp(argv[i], "--descriptor_size")) {
                i = i + 1;
                if (i >= argc) {
                    cerr << "Error introducing input options!!" << endl;
                    return -1;
                } else {
                    options.descriptor_size = atoi(argv[i]);

                    if (options.descriptor_size < 0) {
                        options.descriptor_size = 0;
                    }
                }
            } else if (!strcmp(argv[i], "--show_results")) {
                i = i + 1;
                if (i >= argc) {
                    cerr << "Error introducing input options!!" << endl;
                    return -1;
                } else {
                    options.show_results = (bool) atoi(argv[i]);
                }
            } else if (!strcmp(argv[i], "--verbose")) {
                options.verbosity = true;
            } else if (!strcmp(argv[i],"--output")) {
                i = i+1;
                if (i >= argc) {
                    cerr << "Error introducing input options!!" << endl;
                    return -1;
                }
                else
                    inliers_path = argv[i];
            } else if (!strncmp(argv[i], "--", 2))
                cerr << "Unknown command " << argv[i] << endl;
        }
    } else {
        cerr << "Error introducing input options!!" << endl;
        show_input_options_help(1);
        return -1;
    }

    return 0;
}
