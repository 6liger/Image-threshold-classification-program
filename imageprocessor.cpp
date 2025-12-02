#include "imageprocessor.h"
#include <QDebug>
#include <QPixmap>
#include <cmath>
#include <algorithm>
#include <vector>

ImageProcessor::ImageProcessor(QObject *parent)
    : QObject(parent)
{
}

void ImageProcessor::setImage(const QString& imagePath)
{
    originalImage = cv::imread(imagePath.toStdString());
    results.clear();
}

void ImageProcessor::setParameters(const ProcessingParameters& params)
{
    parameters = params;
}

QPixmap ImageProcessor::matToQPixmap(const cv::Mat& mat)
{
    if (mat.empty()) return QPixmap();

    cv::Mat display;
    if (mat.channels() == 1) {
        cv::cvtColor(mat, display, cv::COLOR_GRAY2RGB);
    } else if (mat.channels() == 3) {
        cv::cvtColor(mat, display, cv::COLOR_BGR2RGB);
    } else {
        display = mat.clone();
    }

    QImage qimg(display.data, display.cols, display.rows, display.step, QImage::Format_RGB888);
    return QPixmap::fromImage(qimg);
}

// 自定义高斯滤波实现
cv::Mat ImageProcessor::customGaussianBlur(const cv::Mat& src, int kernelSize, double sigma)
{
    cv::Mat dst = cv::Mat::zeros(src.size(), src.type());

    // 生成一维高斯核
    int halfSize = kernelSize / 2;
    std::vector<double> kernel1d(kernelSize);
    double sum = 0.0;

    for (int i = -halfSize; i <= halfSize; ++i) {
        double value = exp(-(i * i) / (2.0 * sigma * sigma));
        kernel1d[i + halfSize] = value;
        sum += value;
    }

    // 归一化
    for (int i = 0; i < kernelSize; ++i) {
        kernel1d[i] /= sum;
    }

    // 中间结果存储
    cv::Mat temp = cv::Mat::zeros(src.size(), CV_32F);

    // 水平方向滤波
    for (int y = 0; y < src.rows; ++y) {
        const uchar* srcRow = src.ptr<uchar>(y);
        float* tempRow = temp.ptr<float>(y);

        for (int x = halfSize; x < src.cols - halfSize; ++x) {
            double filteredValue = 0.0;
            for (int kx = -halfSize; kx <= halfSize; ++kx) {
                filteredValue += srcRow[x + kx] * kernel1d[kx + halfSize];
            }
            tempRow[x] = static_cast<float>(filteredValue);
        }

        // 边界处理：复制边界像素
        for (int x = 0; x < halfSize; ++x) {
            tempRow[x] = tempRow[halfSize];
        }
        for (int x = src.cols - halfSize; x < src.cols; ++x) {
            tempRow[x] = tempRow[src.cols - halfSize - 1];
        }
    }

    // 垂直方向滤波
    for (int y = halfSize; y < src.rows - halfSize; ++y) {
        uchar* dstRow = dst.ptr<uchar>(y);

        for (int x = 0; x < src.cols; ++x) {
            double filteredValue = 0.0;
            for (int ky = -halfSize; ky <= halfSize; ++ky) {
                filteredValue += temp.at<float>(y + ky, x) * kernel1d[ky + halfSize];
            }
            dstRow[x] = cv::saturate_cast<uchar>(filteredValue);
        }
    }

    // 垂直边界处理
    for (int y = 0; y < halfSize; ++y) {
        memcpy(dst.ptr<uchar>(y), dst.ptr<uchar>(halfSize), src.cols);
    }
    for (int y = src.rows - halfSize; y < src.rows; ++y) {
        memcpy(dst.ptr<uchar>(y), dst.ptr<uchar>(src.rows - halfSize - 1), src.cols);
    }

    return dst;
}

// 自定义中值滤波实现
cv::Mat ImageProcessor::customMedianBlur(const cv::Mat& src, int kernelSize)
{
    cv::Mat dst = cv::Mat::zeros(src.size(), src.type());
    int halfSize = kernelSize / 2;
    int totalElements = kernelSize * kernelSize;

    // 预分配vector
    std::vector<uchar> values;
    values.reserve(totalElements);

    for (int y = 0; y < src.rows; ++y) {
        const uchar* srcRow = src.ptr<uchar>(y);
        uchar* dstRow = dst.ptr<uchar>(y);

        for (int x = 0; x < src.cols; ++x) {
            values.clear();

            // 确定实际的邻域范围
            int yStart = std::max(0, y - halfSize);
            int yEnd = std::min(src.rows - 1, y + halfSize);
            int xStart = std::max(0, x - halfSize);
            int xEnd = std::min(src.cols - 1, x + halfSize);

            // 收集邻域像素值
            for (int ky = yStart; ky <= yEnd; ++ky) {
                const uchar* neighborRow = src.ptr<uchar>(ky);
                for (int kx = xStart; kx <= xEnd; ++kx) {
                    values.push_back(neighborRow[kx]);
                }
            }

            // 中值查找
            size_t medianIdx = values.size() / 2;
            std::nth_element(values.begin(), values.begin() + medianIdx, values.end());
            dstRow[x] = values[medianIdx];
        }
    }

    return dst;
}

// 大津法阈值计算
int ImageProcessor::computeOtsuThreshold(const cv::Mat& gray)
{
    int hist[256] = { 0 };
    for (int y = 0; y < gray.rows; ++y)
        for (int x = 0; x < gray.cols; ++x)
            ++hist[gray.at<uchar>(y, x)];

    int total = gray.rows * gray.cols;
    double sum = 0;
    for (int i = 0; i < 256; ++i) sum += i * hist[i];

    double sumB = 0, varMax = 0;
    int wB = 0, threshold = 0;

    for (int t = 0; t < 256; ++t) {
        wB += hist[t];
        if (wB == 0) continue;
        int wF = total - wB;
        if (wF == 0) break;

        sumB += t * hist[t];
        double mB = sumB / wB;
        double mF = (sum - sumB) / wF;
        double varBetween = (double)wB * wF * (mB - mF) * (mB - mF);

        if (varBetween > varMax) {
            varMax = varBetween;
            threshold = t;
        }
    }
    return threshold;
}

// 迭代阈值计算
int ImageProcessor::computeIterativeThreshold(const cv::Mat& gray, double eps, int maxIter)
{
    double T = cv::mean(gray)[0];
    for (int i = 0; i < maxIter; ++i) {
        cv::Mat lowMask = gray <= T;
        cv::Mat highMask = gray > T;
        if (cv::countNonZero(lowMask) == 0 || cv::countNonZero(highMask) == 0)
            break;

        double m0 = cv::mean(gray, lowMask)[0];
        double m1 = cv::mean(gray, highMask)[0];
        double newT = 0.5 * (m0 + m1);
        if (std::abs(newT - T) < eps) {
            T = newT;
            break;
        }
        T = newT;
    }
    return static_cast<int>(std::round(T));
}

// 光照估计
cv::Mat ImageProcessor::estimateIlluminationBilateral(const cv::Mat& src)
{
    cv::Mat illumination;
    cv::bilateralFilter(src, illumination, 15, 80, 80);
    cv::GaussianBlur(illumination, illumination, cv::Size(31, 31), 0);
    return illumination;
}

// 光照校正
cv::Mat ImageProcessor::correctIllumination(const cv::Mat& src, const cv::Mat& illumination)
{
    cv::Mat srcFloat, illumFloat;
    src.convertTo(srcFloat, CV_32F);
    illumination.convertTo(illumFloat, CV_32F);

    cv::Mat corrected;
    cv::divide(srcFloat * 180, illumFloat + 5, corrected);

    cv::Mat clamped;
    cv::min(corrected, 255.0, clamped);
    cv::max(clamped, 0.0, clamped);

    cv::Mat result;
    clamped.convertTo(result, CV_8U);

    return result;
}

// 添加文本标签
cv::Mat ImageProcessor::addTextLabel(const cv::Mat& img, const QString& label)
{
    cv::Mat labeled;
    if (img.channels() == 1)
        cv::cvtColor(img, labeled, cv::COLOR_GRAY2BGR);
    else
        labeled = img.clone();

    cv::putText(labeled, label.toStdString(), cv::Point(20, 40),
                cv::FONT_HERSHEY_SIMPLEX, 1.0,
                cv::Scalar(255, 255, 255), 2);
    return labeled;
}

// 绘制直方图
cv::Mat ImageProcessor::drawHistogramBar(const cv::Mat& gray, const QString& title, const cv::Scalar& barColor)
{
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    cv::Mat hist;
    cv::calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);

    double maxVal;
    cv::minMaxLoc(hist, nullptr, &maxVal);

    int histW = 768, histH = 400, margin = 50;
    int binW = cvRound((double)(histW - 2 * margin) / histSize);

    cv::Mat histImage(histH + 80, histW, CV_8UC3, cv::Scalar(255, 255, 255));

    for (int i = 0; i < histSize; ++i) {
        float binVal = hist.at<float>(i);
        int height = cvRound((binVal / maxVal) * histH);
        cv::rectangle(histImage,
                      cv::Point(margin + i * binW, histH - height),
                      cv::Point(margin + (i + 1) * binW - 1, histH),
                      barColor, cv::FILLED);
    }

    cv::line(histImage, cv::Point(margin, histH), cv::Point(histW - margin, histH), cv::Scalar(0, 0, 0), 1);
    for (int i = 0; i <= 4; ++i) {
        int xVal = i * 64;
        int x = margin + xVal * binW;
        cv::putText(histImage, std::to_string(xVal), cv::Point(x - 10, histH + 20),
                    cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0, 0, 0), 1);
        cv::line(histImage, cv::Point(x, histH), cv::Point(x, histH + 5), cv::Scalar(0, 0, 0), 1);
    }
    cv::line(histImage, cv::Point(margin, histH), cv::Point(margin, 20), cv::Scalar(0, 0, 0), 1);
    cv::putText(histImage, title.toStdString(), cv::Point(margin + 10, 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
    return histImage;
}

// 主要处理函数
void ImageProcessor::processImage()
{
    if (originalImage.empty()) {
        emit processingFinished("错误: 没有加载图像");
        return;
    }

    results.clear();
    QString summary;
    int step = 0, totalSteps = 0;

    // 计算总步数
    if (parameters.enableSingleChannel) totalSteps += 8;
    if (parameters.enableDualChannel) totalSteps += 8;
    if (parameters.enableFiltering) totalSteps += 6;
    if (parameters.enableIllumination) totalSteps += 3;
    if (parameters.enableBayesian) totalSteps += 4;
    if (parameters.enableFisherSingle) totalSteps += 3;
    if (parameters.enableFisherMulti) totalSteps += 3;

    emit progressUpdated(0);

    // 分离通道
    std::vector<cv::Mat> channels;
    cv::split(originalImage, channels);
    cv::Mat red = channels[2];
    cv::Mat green = channels[1];
    cv::Mat blue = channels[0];

    cv::Mat trainingMask;
    bool hasTrainingMask = false;
    auto setTrainingMask = [&](const cv::Mat& candidate) {
        if (hasTrainingMask || candidate.empty())
            return;
        cv::Mat normalized;
        if (candidate.type() == CV_8U) {
            normalized = candidate.clone();
        } else {
            candidate.convertTo(normalized, CV_8U);
        }
        hasTrainingMask = !normalized.empty();
        if (hasTrainingMask) {
            trainingMask = normalized;
        }
    };
    auto ensureTrainingMask = [&](const cv::Mat& reference) -> bool {
        if (hasTrainingMask)
            return true;
        if (reference.empty())
            return false;
        cv::Mat autoMask;
        int autoThresh = computeOtsuThreshold(reference);
        cv::threshold(reference, autoMask, autoThresh, 255, cv::THRESH_BINARY);
        setTrainingMask(autoMask);
        return hasTrainingMask;
    };

    // 保存原图和通道
    results.append({"原始图像", matToQPixmap(originalImage)});
    results.append({"红色通道", matToQPixmap(red)});
    results.append({"绿色通道", matToQPixmap(green)});
    results.append({"蓝色通道", matToQPixmap(blue)});

    summary += QString("图像尺寸: %1x%2\n").arg(originalImage.cols).arg(originalImage.rows);

    if (parameters.enableSingleChannel) {
        // 单通道处理
        emit progressUpdated((++step * 100) / totalSteps);

        // 人工阈值
        cv::Mat maskManual;
        cv::threshold(red, maskManual, parameters.manualThreshold, 255, cv::THRESH_BINARY);
        setTrainingMask(maskManual);
        results.append({QString("手动阈值 (T=%1)").arg(parameters.manualThreshold),
                        matToQPixmap(maskManual)});

        // 最优阈值法
        int iterThresh = computeIterativeThreshold(red);
        cv::Mat maskIter;
        cv::threshold(red, maskIter, iterThresh, 255, cv::THRESH_BINARY);
        results.append({QString("最优阈值 (T=%1)").arg(iterThresh),
                        matToQPixmap(maskIter)});

        // 大津法
        int otsuThresh = computeOtsuThreshold(red);
        cv::Mat maskOtsu;
        cv::threshold(red, maskOtsu, otsuThresh, 255, cv::THRESH_BINARY);
        results.append({QString("大津阈值 (T=%1)").arg(otsuThresh),
                        matToQPixmap(maskOtsu)});

        summary += QString("单通道阈值: 手动=%1, 最优=%2, 大津=%3\n")
                       .arg(parameters.manualThreshold).arg(iterThresh).arg(otsuThresh);
    }

    if (parameters.enableDualChannel) {
        emit progressUpdated((++step * 100) / totalSteps);

        // 双通道处理
        cv::Mat dualCombined;
        cv::addWeighted(red, 0.6, green, 0.4, 0, dualCombined);
        results.append({"双通道组合 (0.6R+0.4G)", matToQPixmap(dualCombined)});

        int dualOtsu = computeOtsuThreshold(dualCombined);
        cv::Mat dualMask;
        cv::threshold(dualCombined, dualMask, dualOtsu, 255, cv::THRESH_BINARY);
        results.append({QString("双通道大津 (T=%1)").arg(dualOtsu),
                        matToQPixmap(dualMask)});

        // R-G差值
        cv::Mat diffRg = red - green;
        cv::Mat diffDisp;
        cv::convertScaleAbs(diffRg, diffDisp);
        results.append({"R-G差值图", matToQPixmap(diffDisp)});

        int diffOtsu = computeOtsuThreshold(diffDisp);
        cv::Mat diffMask;
        cv::threshold(diffDisp, diffMask, diffOtsu, 255, cv::THRESH_BINARY);
        results.append({QString("R-G差值大津 (T=%1)").arg(diffOtsu),
                        matToQPixmap(diffMask)});

        summary += QString("双通道阈值: 组合大津=%1, 差值大津=%2\n")
                       .arg(dualOtsu).arg(diffOtsu);
    }

    if (parameters.enableFiltering) {
        emit progressUpdated((++step * 100) / totalSteps);

        // 滤波处理
        cv::Mat gaussFiltered = customGaussianBlur(red, parameters.gaussianKernelSize, parameters.gaussianSigma);
        results.append({"高斯滤波结果", matToQPixmap(gaussFiltered)});

        int gaussOtsu = computeOtsuThreshold(gaussFiltered);
        cv::Mat gaussMask;
        cv::threshold(gaussFiltered, gaussMask, gaussOtsu, 255, cv::THRESH_BINARY);
        results.append({QString("高斯滤波+大津 (T=%1)").arg(gaussOtsu),
                        matToQPixmap(gaussMask)});

        cv::Mat medianFiltered = customMedianBlur(red, parameters.medianKernelSize);
        results.append({"中值滤波结果", matToQPixmap(medianFiltered)});

        int medianOtsu = computeOtsuThreshold(medianFiltered);
        cv::Mat medianMask;
        cv::threshold(medianFiltered, medianMask, medianOtsu, 255, cv::THRESH_BINARY);
        results.append({QString("中值滤波+大津 (T=%1)").arg(medianOtsu),
                        matToQPixmap(medianMask)});

        summary += QString("滤波阈值: 高斯+大津=%1, 中值+大津=%2\n")
                       .arg(gaussOtsu).arg(medianOtsu);
    }

    if (parameters.enableIllumination) {
        emit progressUpdated((++step * 100) / totalSteps);

        // 光照校正
        cv::Mat illumination = estimateIlluminationBilateral(red);
        cv::Mat corrected = correctIllumination(red, illumination);
        results.append({"光照估计", matToQPixmap(illumination)});
        results.append({"光照校正结果", matToQPixmap(corrected)});

        int correctedOtsu = computeOtsuThreshold(corrected);
        cv::Mat correctedMask;
        cv::threshold(corrected, correctedMask, correctedOtsu, 255, cv::THRESH_BINARY);
        results.append({QString("光照校正+大津 (T=%1)").arg(correctedOtsu),
                        matToQPixmap(correctedMask)});

        summary += QString("光照校正阈值: %1\n").arg(correctedOtsu);
    }

    if (parameters.enableBayesian) {
        emit progressUpdated((++step * 100) / totalSteps);

        if (!ensureTrainingMask(red)) {
            summary += "贝叶斯分类: 缺少训练掩膜，跳过\n";
        } else {
            BayesianClassifier classifier;
            classifier.setBasicBGROnly(parameters.bayesUseBasicBGR);
            classifier.setUseFeatureSelection(parameters.bayesUseFeatureSelection);
            classifier.setAdvancedFeatures(parameters.bayesUseAdvancedFeatures);
            classifier.setUseGaussianSmoothing(parameters.bayesUseGaussianSmoothing);

            int featureCount = std::max(1, parameters.bayesFeatureCount);
            if (!classifier.train(originalImage, trainingMask, featureCount)) {
                summary += "贝叶斯分类: 训练失败\n";
            } else {
                cv::Mat bayesMask = classifier.classify(originalImage);
                results.append({"贝叶斯分类掩膜", matToQPixmap(bayesMask)});

                cv::Mat confusion = BayesianClassifier::visualizeConfusionMatrix(bayesMask, trainingMask);
                if (!confusion.empty()) {
                    results.append({"贝叶斯混淆可视化", matToQPixmap(confusion)});
                }

                double acc = 0, prec = 0, recall = 0, f1 = 0;
                BayesianClassifier::calculateMetrics(bayesMask, trainingMask, acc, prec, recall, f1);
                summary += QString("贝叶斯分类: Acc=%1, F1=%2, 选用特征=%3\n")
                               .arg(acc, 0, 'f', 3)
                               .arg(f1, 0, 'f', 3)
                               .arg(classifier.getSelectedFeatures().size());

                if (parameters.bayesUseFeatureSelection) {
                    auto gains = classifier.getFeatureGains(originalImage, trainingMask);
                    cv::Mat gainViz = visualizeInformationGain(gains, classifier.getSelectedFeatures());
                    if (!gainViz.empty()) {
                        results.append({"特征信息增益", matToQPixmap(gainViz)});
                    }
                }
            }
        }
    }

    if (parameters.enableFisherSingle) {
        emit progressUpdated((++step * 100) / totalSteps);

        if (!ensureTrainingMask(red)) {
            summary += "Fisher单通道: 缺少训练掩膜，跳过\n";
        } else {
            FisherClassifier fisherSingle;
            fisherSingle.setEnableMorphology(parameters.fisherUseMorphology);
            fisherSingle.setMorphologySize(parameters.fisherMorphSize);

            int channel = std::clamp(parameters.fisherSingleChannel, 0, 2);
            if (!fisherSingle.trainSingleBand(originalImage, trainingMask, channel)) {
                summary += "Fisher单通道: 训练失败\n";
            } else {
                static const char* channelNames[] = { "蓝", "绿", "红" };
                cv::Mat fisherMask = fisherSingle.classifySingleBand(originalImage, channel);
                results.append({QString("Fisher单通道(%1)").arg(channelNames[channel]),
                                matToQPixmap(fisherMask)});

                summary += QString("Fisher单通道: Fisher比率=%1\n")
                               .arg(fisherSingle.getFisherRatio(), 0, 'f', 3);

                cv::Mat distribution = fisherSingle.visualizeProjectionDistribution(originalImage, trainingMask);
                if (!distribution.empty()) {
                    results.append({"Fisher单通道投影", matToQPixmap(distribution)});
                }
            }
        }
    }

    if (parameters.enableFisherMulti) {
        emit progressUpdated((++step * 100) / totalSteps);

        if (!ensureTrainingMask(red)) {
            summary += "Fisher多通道: 缺少训练掩膜，跳过\n";
        } else {
            FisherClassifier fisherMulti;
            fisherMulti.setEnableMorphology(parameters.fisherUseMorphology);
            fisherMulti.setMorphologySize(parameters.fisherMorphSize);

            if (!fisherMulti.trainMultiBand(originalImage, trainingMask)) {
                summary += "Fisher多通道: 训练失败\n";
            } else {
                cv::Mat fisherMask = fisherMulti.classifyMultiBand(originalImage);
                results.append({"Fisher多通道", matToQPixmap(fisherMask)});

                summary += QString("Fisher多通道: Fisher比率=%1\n")
                               .arg(fisherMulti.getFisherRatio(), 0, 'f', 3);

                cv::Mat distribution = fisherMulti.visualizeProjectionDistribution(originalImage, trainingMask);
                if (!distribution.empty()) {
                    results.append({"Fisher多通道投影", matToQPixmap(distribution)});
                }
            }
        }
    }

    emit progressUpdated(100);
    emit processingFinished(summary);
}
