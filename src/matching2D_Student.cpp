
#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
  // configure matcher
  bool crossCheck = false;
  cv::Ptr<cv::DescriptorMatcher> matcher;

  if (matcherType.compare("MAT_BF") == 0)
  {
    int normType = cv::NORM_HAMMING;
    matcher = cv::BFMatcher::create(normType, crossCheck);
  }
  else if (matcherType.compare("MAT_FLANN") == 0)
  {
    if (descSource.type() != CV_32F)
    { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
      descSource.convertTo(descSource, CV_32F);
      descRef.convertTo(descRef, CV_32F);
    }
    
    matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
  }

  // perform matching task
  if (selectorType.compare("SEL_NN") == 0)
  { // nearest neighbor (best match)

    matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
  }
  else if (selectorType.compare("SEL_KNN") == 0)
  { // k nearest neighbors (k=2)

    vector<vector<cv::DMatch>> matches_vect;
    double t = (double)cv::getTickCount();
    matcher->knnMatch(descSource, descRef, matches_vect, 2); // Finds the best match for each descriptor in desc1

    // filter matches using descriptor distance ratio test
    for (auto m_it = matches_vect.begin(); m_it != matches_vect.end(); ++m_it)
    {
      if (m_it->at(0).distance / m_it->at(1).distance < 0.8)
        matches.push_back(m_it->at(0));
    }
  }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
  // select appropriate descriptor
  cv::Ptr<cv::DescriptorExtractor> extractor;
  if (descriptorType.compare("BRISK") == 0)
  {

    int threshold = 30;    // FAST/AGAST detection threshold score.
    int octaves = 3;       // detection octaves (use 0 to do single scale)
    float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

    extractor = cv::BRISK::create(threshold, octaves, patternScale);
  }
  else if (descriptorType.compare("BRIEF") == 0)
  {
    int bytes = 32;
		bool use_orientation = false;
    
    extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(bytes, use_orientation); 		
  }
  else if (descriptorType.compare("ORB") == 0)
  {
    int nfeatures = 500;
		float scaleFactor = 1.2f;
		int nlevels = 8;
		int edgeThreshold = 31;
		int firstLevel = 0;
		int WTA_K = 2;
		cv::ORB::ScoreType scoreType = cv::ORB::HARRIS_SCORE;
		int patchSize = 31;
		int fastThreshold = 20;

    extractor = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold,
                                firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
  }
  else if (descriptorType.compare("FREAK") == 0)
  {
    bool orientationNormalized = true;
		bool scaleNormalized = true;
		float patternScale = 22.0f;
		int nOctaves = 4;
		const std::vector<int>& selectedPairs = std::vector<int>();

    extractor = cv::xfeatures2d::FREAK::create(orientationNormalized, scaleNormalized,
                                               patternScale, nOctaves, selectedPairs); 		
  }
  else if (descriptorType.compare("AKAZE") == 0)
  {
    cv::AKAZE::DescriptorType descriptorType = cv::AKAZE::DESCRIPTOR_MLDB;
		int descriptorSize = 0;
		int descriptorChannels = 3;
		float threshold = 0.001f;
		int nOctaves = 4;
		int nOctaveLayers = 4;
		cv::KAZE::DiffusivityType diffusivity = cv::KAZE::DIFF_PM_G2;

    extractor = cv::AKAZE::create(descriptorType, descriptorSize, descriptorChannels,
                                  threshold,	nOctaves,	nOctaveLayers, diffusivity); 
  }
  else if (descriptorType.compare("SIFT") == 0)
  {
    int	nfeatures = 0;
		int	nOctaveLayers = 3;
		double contrastThreshold = 0.04;
		double edgeThreshold = 10;
		double sigma = 1.6;

    extractor = cv::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold,
                                 edgeThreshold, sigma);
  }

  // perform feature description
  double t = (double)cv::getTickCount();
  extractor->compute(img, keypoints, descriptors);
  t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
  cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
  // compute detector parameters based on image size
  int blockSize = 4;       // size of an average block for computing a derivative covariation matrix over each pixel neighborhood
  double maxOverlap = 0.0; // max. permissible overlap between two features in %
  double minDistance = (1.0 - maxOverlap) * blockSize;
  int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

  double qualityLevel = 0.01; // minimal accepted quality of image corners
  double k = 0.04;

  // Apply corner detection
  double t = (double)cv::getTickCount();
  vector<cv::Point2f> corners;
  cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

  // add corners to result vector
  for (auto it = corners.begin(); it != corners.end(); ++it)
  {

    cv::KeyPoint newKeyPoint;
    newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
    newKeyPoint.size = blockSize;
    keypoints.push_back(newKeyPoint);
  }
  t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
  cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

  // visualize results
  if (bVis)
  {
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    string windowName = "Shi-Tomasi Corner Detector Results";
    cv::namedWindow(windowName, 6);
    imshow(windowName, visImage);
    cv::waitKey(0);
  }
}

// Detect keypoints in image using the traditional Harris detector
void detKeypointsHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
  // detector parameters
  int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
  int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
  int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
  double k = 0.04;       // Harris parameter (see equation for details)

  // Apply corner detection
  double t = (double)cv::getTickCount();
  
  cv::Mat dst, dstNorm;
  dst = cv::Mat::zeros(img.size(), CV_32FC1);
  cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
  cv::normalize(dst, dstNorm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

  for (int r = 0; r < dstNorm.rows; r++)
  {
    for (int c = 0; c < dstNorm.cols; c++)
    {
      if (dstNorm.at<float>(r, c) >= minResponse)
      {
        cv::KeyPoint newKeypoint;
        newKeypoint.pt = cv::Point2f(c, r);
        newKeypoint.size = 2 * apertureSize;
        newKeypoint.response = dstNorm.at<float>(r, c);
                
        bool suppressed = false;
        for (auto kIt = keypoints.begin(); kIt != keypoints.end(); )
        {
          if (cv::KeyPoint::overlap(newKeypoint, *kIt) > 0)
          {
            if (newKeypoint.response <= kIt->response)
            {
              suppressed = true;
              break;
            }
            else
            {
              kIt = keypoints.erase(kIt);
            }
          }
          else
          {
            ++kIt;
          }
        }

        if (!suppressed)
        {
          keypoints.push_back(newKeypoint);
        }
      }
    }
  }

  t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
  cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

  // visualize results
  if (bVis)
  {
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    string windowName = "Harris Corner Detector Results";
    cv::namedWindow(windowName, 6);
    imshow(windowName, visImage);
    cv::waitKey(0);
  }
}

// Detect keypoints in image using the modern keypoint detectors
void detKeypointsModern(vector<cv::KeyPoint> &keypoints, cv::Mat &img, string detectorType, bool bVis)
{
  cv::Ptr<cv::FeatureDetector> detector;

  if (detectorType.compare("FAST") == 0)
  {
    int threshold = 45;
    bool nonMaxSupp = true;
    int type = cv::FastFeatureDetector::TYPE_9_16;
    detector = cv::FastFeatureDetector::create(threshold, nonMaxSupp);
  }
  else if (detectorType.compare("BRISK") == 0)
  {
    int threshold = 30;
    int octaves = 3;
    float patternScale = 1.0f;
    detector = cv::BRISK::create(threshold, octaves, patternScale);
  }
  else if (detectorType.compare("ORB") == 0)
  {
    int nfeatures = 500;
		float scaleFactor = 1.2f;
		int nlevels = 8;
		int edgeThreshold = 31;
		int firstLevel = 0;
		int WTA_K = 2;
		cv::ORB::ScoreType scoreType = cv::ORB::HARRIS_SCORE;
		int patchSize = 31;
		int fastThreshold = 20; 
    detector = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold,
                               firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
  }
  else if (detectorType.compare("AKAZE") == 0)
  {
    cv::AKAZE::DescriptorType descriptorType = cv::AKAZE::DESCRIPTOR_MLDB;
		int descriptorSize = 0;
		int descriptorChannels = 3;
		float threshold = 0.001f;
		int nOctaves = 4;
		int nOctaveLayers = 4;
		cv::KAZE::DiffusivityType diffusivity = cv::KAZE::DIFF_PM_G2;
    detector = cv::AKAZE::create(descriptorType, descriptorSize, descriptorChannels,
                                 threshold,	nOctaves,	nOctaveLayers, diffusivity); 
	}
  else if (detectorType.compare("SIFT") == 0)
  {
    int	nfeatures = 0;
		int	nOctaveLayers = 3;
		double contrastThreshold = 0.04;
		double edgeThreshold = 10;
		double sigma = 1.6; 
    detector = cv::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold,
                                edgeThreshold, sigma);
  }

  double t = (double)cv::getTickCount();
  detector->detect(img, keypoints);
  t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
  cout << detectorType << " detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;


  if (bVis)
  {
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    stringstream windowName;
    windowName << detectorType << " Corner Detector Results";
    cv::namedWindow(windowName.str(), 6);
    imshow(windowName.str(), visImage);
    cv::waitKey(0);
  }
}