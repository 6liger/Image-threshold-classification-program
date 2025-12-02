#ifndef BAYESIAN_CLASSIFIER_H
#define BAYESIAN_CLASSIFIER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <set>

class BayesianClassifier
{
public:
    void setBasicBGROnly(bool value);
    void setUseFeatureSelection(bool value);
    void setAdvancedFeatures(bool value);
    void setUseGaussianSmoothing(bool value);
    void setExtraFeatureImage(const cv::Mat& extraFeature);

    std::vector<int> getSelectedFeatures() const;
    std::vector<double> getFeatureGains(const cv::Mat& image, const cv::Mat& mask);

    bool train(const cv::Mat& image, const cv::Mat& mask, int k = 8);
    cv::Mat classify(const cv::Mat& image);
    cv::Mat classify(const cv::Mat& image, const cv::Mat& extraFeature);

    static void calculateMetrics(const cv::Mat& result, const cv::Mat& groundTruth,
                                 double& accuracy, double& precision,
                                 double& recall, double& f1);
    static cv::Mat visualizeConfusionMatrix(const cv::Mat& result, const cv::Mat& groundTruth);

private:
    cv::Mat extractFeatures(const cv::Mat& image);
    std::vector<double> calculateInformationGain(const cv::Mat& features, const cv::Mat& labels);
    cv::Mat classifyWithExtraFeature(const cv::Mat& image, const cv::Mat& extraFeature);

    bool useBasicBGROnly = false;
    bool useFeatureSelection = true;
    bool useAdvancedFeatures = true;
    bool useGaussianSmoothing = false;
    bool hasExtraFeature = false;

    cv::Mat extraFeatureImage;
    std::vector<int> selectedFeatures;
    std::vector<cv::Mat> means;
    std::vector<cv::Mat> stdDevs;
    std::vector<double> priorProbs;
};

cv::Mat visualizeInformationGain(const std::vector<double>& gains,
                                 const std::vector<int>& selectedFeatures);
cv::Mat optimizedBayesClassify(const cv::Mat& image, const cv::Mat& mask);
cv::Mat bayesFisherHybridClassify(const cv::Mat& image, const cv::Mat& mask);

#endif // BAYESIAN_CLASSIFIER_H

