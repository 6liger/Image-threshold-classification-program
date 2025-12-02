#ifndef IMAGEPROCESSOR_H
#define IMAGEPROCESSOR_H

#include <QObject>
#include <QThread>
#include <QString>
#include <QPixmap>
#include <opencv2/opencv.hpp>

#include "bayesian_classifier.h"
#include "fisher_classifier.h"

struct ProcessingParameters {
    int manualThreshold = 170;
    int gaussianKernelSize = 5;
    double gaussianSigma = 1.5;
    int medianKernelSize = 5;

    bool enableSingleChannel = true;
    bool enableDualChannel = true;
    bool enableFiltering = true;
    bool enableIllumination = true;
    bool enableBayesian = false;
    bool enableFisherSingle = false;
    bool enableFisherMulti = false;

    bool bayesUseBasicBGR = false;
    bool bayesUseAdvancedFeatures = true;
    bool bayesUseFeatureSelection = true;
    bool bayesUseGaussianSmoothing = true;
    int bayesFeatureCount = 8;

    bool fisherUseMorphology = true;
    int fisherMorphSize = 5;
    int fisherSingleChannel = 2; // 0:B,1:G,2:R
};

class ImageProcessor : public QObject
{
    Q_OBJECT

public:
    explicit ImageProcessor(QObject *parent = nullptr);

    void setImage(const QString& imagePath);
    void setParameters(const ProcessingParameters& params);
    void processImage();

    QList<QPair<QString, QPixmap>> getResults() const { return results; }

signals:
    void processingFinished(const QString& summary);
    void progressUpdated(int percentage);

private:
    // 原有的处理函数
    cv::Mat customGaussianBlur(const cv::Mat& src, int kernelSize, double sigma);
    cv::Mat customMedianBlur(const cv::Mat& src, int kernelSize);
    int computeOtsuThreshold(const cv::Mat& gray);
    int computeIterativeThreshold(const cv::Mat& gray, double eps = 0.5, int maxIter = 100);
    cv::Mat estimateIlluminationBilateral(const cv::Mat& src);
    cv::Mat correctIllumination(const cv::Mat& src, const cv::Mat& illumination);
    cv::Mat addTextLabel(const cv::Mat& img, const QString& label);
    cv::Mat drawHistogramBar(const cv::Mat& gray, const QString& title, const cv::Scalar& barColor);

    // 转换函数
    QPixmap matToQPixmap(const cv::Mat& mat);

    cv::Mat originalImage;
    ProcessingParameters parameters;
    QList<QPair<QString, QPixmap>> results;
};

#endif // IMAGEPROCESSOR_H
