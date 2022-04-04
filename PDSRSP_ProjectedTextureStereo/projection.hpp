#pragma once
#include <opencv2/core.hpp>

void projectPattern(const cv::Mat& leftImage, cv::Mat& destLeftImage, const cv::Mat& rightImage, cv::Mat& destRightImage, const cv::Mat& centerDisparityMap32F, const cv::Mat& pattern, const float mask_size, const float dark_amp);
void addRandmizedSatellitePoints(const cv::Mat& mask, cv::Mat& dest, cv::RNG& rng);
void loadPrecomputedPDS(cv::Mat& mask, int distanceIndex, int index);