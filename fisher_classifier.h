#ifndef FISHER_CLASSIFIER_H
#define FISHER_CLASSIFIER_H

#include <opencv2/opencv.hpp>

class FisherClassifier
{
public:
    void setEnableMorphology(bool enable);
    void setMorphologySize(int size);

    bool trainSingleBand(const cv::Mat& image, const cv::Mat& mask, int channel);
    bool trainMultiBand(const cv::Mat& image, const cv::Mat& mask);

    cv::Mat classifySingleBand(const cv::Mat& image, int channel);
    cv::Mat classifyMultiBand(const cv::Mat& image);

    double getFisherRatio() const;
    cv::Mat visualizeProjectionDistribution(const cv::Mat& image, const cv::Mat& mask);

private:
    bool enableMorphology = true;
    int morphologySize = 5;
    bool isTrained = false;

    double fisherRatio = 0.0;
    double threshold = 0.0;
    cv::Mat projectionVector;
};

#endif // FISHER_CLASSIFIER_H

