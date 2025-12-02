#include "bayesian_classifier.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//��������ģʽ - �Ƿ�ʹ�û���BGR����
void BayesianClassifier::setBasicBGROnly(bool value) {
    useBasicBGROnly = value;
}

//��������ѡ�񿪹�
void BayesianClassifier::setUseFeatureSelection(bool value) {
    useFeatureSelection = value;
}

//�����Ƿ�ʹ�ø߼�����
void BayesianClassifier::setAdvancedFeatures(bool value) {
    useAdvancedFeatures = value;
}

//�����Ƿ�ʹ�ø�˹ƽ��
void BayesianClassifier::setUseGaussianSmoothing(bool value) {
    useGaussianSmoothing = value;
}

//���ö�������ͼ��
void BayesianClassifier::setExtraFeatureImage(const cv::Mat& extraFeature) {
    this->extraFeatureImage = extraFeature.clone();
    hasExtraFeature = true;
}

//��ȡ��ѡ�������
std::vector<int> BayesianClassifier::getSelectedFeatures() const {
    return selectedFeatures;
}

//������ͨ������ͼ
cv::Mat BayesianClassifier::extractFeatures(const cv::Mat& image) {
    try {
        if (image.empty()) {
            std::cerr << "Empty input image" << std::endl;
            return cv::Mat();
        }

        int numChannels = image.channels();
        if (numChannels < 1) {
            std::cerr << "Invalid number of channels in input image" << std::endl;
            return cv::Mat();
        }

        cv::Mat processedImage = image.clone();
        if (useGaussianSmoothing) {
            cv::GaussianBlur(image, processedImage, cv::Size(5, 5), 0);
        }

        if (useBasicBGROnly && !useAdvancedFeatures) {
            //��ʹ��BGRͨ��
            std::vector<cv::Mat> channels;

            if (numChannels == 1) {
                channels.push_back(processedImage);
            }
            else {
                cv::split(processedImage, channels);
            }

            for (int i = 0; i < channels.size(); i++) {
                cv::Mat temp;
                channels[i].convertTo(temp, CV_32F);
                channels[i] = temp;
            }

            if (hasExtraFeature && !extraFeatureImage.empty()) {
                cv::Mat extraFeature;
                extraFeatureImage.convertTo(extraFeature, CV_32F);
                channels.push_back(extraFeature);
            }

            //�ϲ�����ͨ��
            cv::Mat featureImage;
            cv::merge(channels, featureImage);

            return featureImage;
        }
        else {
            //ʹ�ø�������
            cv::Mat bgrImage;
            if (numChannels == 1) {
                cv::cvtColor(processedImage, bgrImage, cv::COLOR_GRAY2BGR);
            }
            else if (numChannels == 3) {
                bgrImage = processedImage.clone();
            }
            else if (numChannels == 4) {
                std::vector<cv::Mat> allChannels;
                cv::split(processedImage, allChannels);
                std::vector<cv::Mat> bgrChannels = { allChannels[0], allChannels[1], allChannels[2] };
                cv::merge(bgrChannels, bgrImage);
            }
            else {
                std::cerr << "Unsupported channel count: " << numChannels << std::endl;
                return cv::Mat();
            }

            cv::Mat hsv, lab;
            cv::cvtColor(bgrImage, hsv, cv::COLOR_BGR2HSV);
            cv::cvtColor(bgrImage, lab, cv::COLOR_BGR2Lab);

            //��ͬ��ɫ����
            std::vector<cv::Mat> channels;
            std::vector<cv::Mat> hsvChannels;
            std::vector<cv::Mat> labChannels;

            cv::split(bgrImage, channels);
            cv::split(hsv, hsvChannels);
            cv::split(lab, labChannels);

            //ȷ������ͨ������Ч
            if (channels.size() != 3 || hsvChannels.size() != 3 || labChannels.size() != 3) {
                std::cerr << "Invalid number of channels" << std::endl;
                return cv::Mat();
            }

            //�����ݶ�����
            cv::Mat grayImage;
            cv::cvtColor(bgrImage, grayImage, cv::COLOR_BGR2GRAY);

            cv::Mat gradX, gradY, gradMagnitude;
            cv::Sobel(grayImage, gradX, CV_32F, 1, 0);
            cv::Sobel(grayImage, gradY, CV_32F, 0, 1);
            cv::magnitude(gradX, gradY, gradMagnitude);

            //����ֲ���������
            cv::Mat localVariance, meanSquared;
            cv::Mat grayFloat;
            grayImage.convertTo(grayFloat, CV_32F);
            cv::boxFilter(grayFloat, localVariance, CV_32F, cv::Size(5, 5));
            cv::Mat squaredImage;
            cv::multiply(grayFloat, grayFloat, squaredImage);
            cv::boxFilter(squaredImage, meanSquared, CV_32F, cv::Size(5, 5));
            cv::subtract(meanSquared, localVariance.mul(localVariance), localVariance);

            //�����������
            std::vector<cv::Mat> featureChannels;

            cv::Mat temp;
            for (int i = 0; i < 3; i++) {
                channels[i].convertTo(temp, CV_32F);
                featureChannels.push_back(temp.clone());

                hsvChannels[i].convertTo(temp, CV_32F);
                featureChannels.push_back(temp.clone());

                labChannels[i].convertTo(temp, CV_32F);
                featureChannels.push_back(temp.clone());
            }

            gradMagnitude.convertTo(temp, CV_32F);
            featureChannels.push_back(temp.clone());

            localVariance.convertTo(temp, CV_32F);
            featureChannels.push_back(temp.clone());

            if (useAdvancedFeatures) {
                cv::Mat lbpFeature = cv::Mat::zeros(grayImage.size(), CV_8U);
                for (int i = 1; i < grayImage.rows - 1; i++) {
                    for (int j = 1; j < grayImage.cols - 1; j++) {
                        uchar center = grayImage.at<uchar>(i, j);
                        unsigned char code = 0;
                        code |= (grayImage.at<uchar>(i - 1, j - 1) > center) << 7;
                        code |= (grayImage.at<uchar>(i - 1, j) > center) << 6;
                        code |= (grayImage.at<uchar>(i - 1, j + 1) > center) << 5;
                        code |= (grayImage.at<uchar>(i, j + 1) > center) << 4;
                        code |= (grayImage.at<uchar>(i + 1, j + 1) > center) << 3;
                        code |= (grayImage.at<uchar>(i + 1, j) > center) << 2;
                        code |= (grayImage.at<uchar>(i + 1, j - 1) > center) << 1;
                        code |= (grayImage.at<uchar>(i, j - 1) > center) << 0;
                        lbpFeature.at<uchar>(i, j) = code;
                    }
                }

                lbpFeature.convertTo(temp, CV_32F);
                featureChannels.push_back(temp.clone());

                //��ɫ���Ͷ�
                cv::Mat saturation = hsvChannels[1].clone();
                cv::Mat satVariance;
                cv::GaussianBlur(saturation, satVariance, cv::Size(9, 9), 0);
                cv::absdiff(saturation, satVariance, satVariance);

                satVariance.convertTo(temp, CV_32F);
                featureChannels.push_back(temp.clone());

                //��ɫ�Աȶ�
                cv::Mat contrast;
                cv::Laplacian(grayImage, contrast, CV_32F);
                cv::convertScaleAbs(contrast, contrast);

                contrast.convertTo(temp, CV_32F);
                featureChannels.push_back(temp.clone());
            }

            if (hasExtraFeature && !extraFeatureImage.empty()) {
                cv::Mat extraFeature;
                extraFeatureImage.convertTo(extraFeature, CV_32F);
                featureChannels.push_back(extraFeature);
            }

            //���
            cv::Mat featureImage;
            cv::merge(featureChannels, featureImage);

            return featureImage;
        }
    }
    catch (const cv::Exception& e) {
        std::cerr << "OpenCV exception in extractFeatures: " << e.what() << std::endl;
        return cv::Mat();
    }
    catch (const std::exception& e) {
        std::cerr << "Exception in extractFeatures: " << e.what() << std::endl;
        return cv::Mat();
    }
}

//����ÿ����������Ϣ����
std::vector<double> BayesianClassifier::calculateInformationGain(const cv::Mat& features, const cv::Mat& labels) {
    int numFeatures = features.cols;
    std::vector<double> gains(numFeatures, 0.0);

    //��ֹ������
    if (features.empty() || labels.empty() || features.rows != labels.rows) {
        std::cerr << "Invalid feature or label data" << std::endl;
        return gains;
    }

    //���������
    std::map<int, int> labelCounts;
    for (int i = 0; i < labels.rows; i++) {
        labelCounts[labels.at<int>(i, 0)]++;
    }

    double classEntropy = 0.0;
    for (const auto& pair : labelCounts) {
        double prob = static_cast<double>(pair.second) / labels.rows;
        if (prob > 0) {  
            classEntropy -= prob * log2(prob);
        }
    }

    for (int f = 0; f < numFeatures; f++) {
        try {
            int bins = 10;
            std::map<int, std::map<int, int>> featureLabelCounts;
            std::map<int, int> featureCounts;

            double minVal = std::numeric_limits<double>::max();
            double maxVal = std::numeric_limits<double>::lowest();

            //�ҵ���������Сֵ�����ֵ
            for (int i = 0; i < features.rows; i++) {
                float val = features.at<float>(i, f);
                if (std::isfinite(val)) {
                    minVal = std::min(minVal, static_cast<double>(val));
                    maxVal = std::max(maxVal, static_cast<double>(val));
                }
            }

            double range = maxVal - minVal;
            if (range <= 0) range = 1.0; 
            for (int i = 0; i < features.rows; i++) {
                float val = features.at<float>(i, f);
                if (!std::isfinite(val)) continue;

                // ������ֵӳ�䵽[0,bins-1]��Χ
                int bin = std::min(bins - 1, std::max(0, static_cast<int>((val - minVal) / range * bins)));
                featureLabelCounts[bin][labels.at<int>(i, 0)]++;
                featureCounts[bin]++;
            }

            double conditionalEntropy = 0.0;
            for (const auto& binPair : featureCounts) {
                int bin = binPair.first;
                double binProb = static_cast<double>(binPair.second) / features.rows;
                if (binProb <= 0) continue; 

                double binEntropy = 0.0;
                for (const auto& labelPair : featureLabelCounts[bin]) {
                    if (featureCounts[bin] > 0) {
                        double labelProb = static_cast<double>(labelPair.second) / featureCounts[bin];
                        if (labelProb > 0) {
                            binEntropy -= labelProb * log2(labelProb);
                        }
                    }
                }

                conditionalEntropy += binProb * binEntropy;
            }

            gains[f] = classEntropy - conditionalEntropy;

            //������Ч��Ϣ����
            if (!std::isfinite(gains[f])) {
                gains[f] = 0.0;
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Exception in calculateInformationGain for feature " << f << ": " << e.what() << std::endl;
            gains[f] = 0.0;
        }
    }

    return gains;
}

//��ȡ������Ϣ����
std::vector<double> BayesianClassifier::getFeatureGains(const cv::Mat& image, const cv::Mat& mask) {
    cv::Mat featureImage = extractFeatures(image);
    int numFeatures = featureImage.channels();

    //׼��ѵ������
    std::vector<cv::Mat> featureChannels;
    cv::split(featureImage, featureChannels);

    cv::Mat foregroundMask = mask > 0;
    cv::Mat backgroundMask = mask == 0;

    int numForegroundPixels = cv::countNonZero(foregroundMask);
    int numBackgroundPixels = cv::countNonZero(backgroundMask);
    int totalPixels = numForegroundPixels + numBackgroundPixels;

    //������������ͱ�ǩ����
    cv::Mat features(totalPixels, numFeatures, CV_32F);
    cv::Mat labels(totalPixels, 1, CV_32S);

    int idx = 0;
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            if (foregroundMask.at<uchar>(i, j) || backgroundMask.at<uchar>(i, j)) {
                for (int f = 0; f < numFeatures; f++) {
                    features.at<float>(idx, f) = featureChannels[f].at<float>(i, j);
                }

                labels.at<int>(idx, 0) = foregroundMask.at<uchar>(i, j) ? 1 : 0;
                idx++;
            }
        }
    }

    //����ʵ��ʹ�õ����ݴ�С
    features = features.rowRange(0, idx);
    labels = labels.rowRange(0, idx);

    return calculateInformationGain(features, labels);
}

bool BayesianClassifier::train(const cv::Mat& image, const cv::Mat& mask, int k) {
    try {
        if (image.empty() || mask.empty()) {
            std::cerr << "Empty input image or mask" << std::endl;
            return false;
        }

        if (image.size() != mask.size()) {
            std::cerr << "Image and mask size mismatch" << std::endl;
            return false;
        }

        //��ȡ����
        cv::Mat featureImage = extractFeatures(image);
        if (featureImage.empty()) {
            std::cerr << "Failed to extract features" << std::endl;
            return false;
        }

        int numFeatures = featureImage.channels();
        std::cout << "Extracted " << numFeatures << " features" << std::endl;

        //׼��ѵ������
        std::vector<cv::Mat> featureChannels;
        cv::split(featureImage, featureChannels);

        cv::Mat foregroundMask = mask > 0;
        cv::Mat backgroundMask = mask == 0;

        int numForegroundPixels = cv::countNonZero(foregroundMask);
        int numBackgroundPixels = cv::countNonZero(backgroundMask);
        int totalPixels = numForegroundPixels + numBackgroundPixels;

        std::cout << "Foreground pixels: " << numForegroundPixels << std::endl;
        std::cout << "Background pixels: " << numBackgroundPixels << std::endl;

        if (numForegroundPixels == 0 || numBackgroundPixels == 0) {
            std::cerr << "Invalid mask: missing foreground or background pixels" << std::endl;
            return false;
        }

        //������������ͱ�ǩ����
        cv::Mat features(totalPixels, numFeatures, CV_32F);
        cv::Mat labels(totalPixels, 1, CV_32S);

        int idx = 0;
        for (int i = 0; i < image.rows; i++) {
            for (int j = 0; j < image.cols; j++) {
                if (idx >= totalPixels) {
                    std::cerr << "Index out of bounds" << std::endl;
                    return false;
                }

                if (foregroundMask.at<uchar>(i, j) || backgroundMask.at<uchar>(i, j)) {
                    //��ȡÿ�����ص���������
                    for (int f = 0; f < numFeatures; f++) {
                        if (f < featureChannels.size()) {
                            features.at<float>(idx, f) = featureChannels[f].at<float>(i, j);
                        }
                    }

                    labels.at<int>(idx, 0) = foregroundMask.at<uchar>(i, j) ? 1 : 0;
                    idx++;
                }
            }
        }

        features = features.rowRange(0, idx);
        labels = labels.rowRange(0, idx);

        std::cout << "Prepared " << idx << " samples for training" << std::endl;

        selectedFeatures.clear();

        if (useBasicBGROnly) {
            int numBGRFeatures = std::min(3, numFeatures);
            for (int i = 0; i < numBGRFeatures; i++) {
                selectedFeatures.push_back(i);
            }
            std::cout << "Using basic BGR channels only: " << selectedFeatures.size() << " features" << std::endl;
        }
        else if (useFeatureSelection && !useBasicBGROnly) {
            //������Ϣ���沢ѡ���������
            std::vector<double> gains = calculateInformationGain(features, labels);
            std::vector<std::pair<double, int>> gainPairs;
            for (int i = 0; i < gains.size(); i++) {
                gainPairs.push_back({ gains[i], i });
            }
            std::sort(gainPairs.begin(), gainPairs.end(), std::greater<std::pair<double, int>>());
            for (int i = 0; i < std::min(k, static_cast<int>(gainPairs.size())); i++) {
                selectedFeatures.push_back(gainPairs[i].second);
                std::cout << "Feature " << gainPairs[i].second << " with gain " << gainPairs[i].first << std::endl;
            }

            std::cout << "Selected " << selectedFeatures.size() << " features based on information gain" << std::endl;
        }
        else {
            for (int i = 0; i < numFeatures; i++) {
                selectedFeatures.push_back(i);
            }
            std::cout << "Using all " << numFeatures << " features (no selection)" << std::endl;
        }

        if (selectedFeatures.empty()) {
            std::cerr << "No features selected" << std::endl;
            return false;
        }

        means.resize(2);
        stdDevs.resize(2);
        priorProbs.resize(2);

        std::vector<cv::Mat> classFeatures(2);
        std::vector<int> classCounts(2);

        //����ǰ���ͱ���
        for (int i = 0; i < features.rows; i++) {
            int label = labels.at<int>(i, 0);
            if (label == 0 || label == 1) {
                if (classFeatures[label].empty()) {
                    classFeatures[label] = cv::Mat(0, features.cols, CV_32F);
                }
                classFeatures[label].push_back(features.row(i));
                classCounts[label]++;
            }
        }

        for (int c = 0; c < 2; c++) {
            means[c] = cv::Mat::zeros(1, numFeatures, CV_32F);
            stdDevs[c] = cv::Mat::zeros(1, numFeatures, CV_32F);
            priorProbs[c] = static_cast<double>(classCounts[c]) / features.rows;

            std::cout << "Class " << c << " has " << classCounts[c] << " samples, prior = " << priorProbs[c] << std::endl;

            if (classFeatures[c].empty()) {
                std::cerr << "No samples for class " << c << std::endl;
                continue;
            }

            for (int f : selectedFeatures) {
                if (f >= 0 && f < numFeatures) {
                    cv::Scalar mean, stddev;
                    cv::meanStdDev(classFeatures[c].col(f), mean, stddev);
                    means[c].at<float>(0, f) = mean[0];
                    stdDevs[c].at<float>(0, f) = stddev[0] + 1e-6; // ��ֹ������
                }
            }
        }

        return true;
    }
    catch (const cv::Exception& e) {
        std::cerr << "OpenCV exception in train: " << e.what() << std::endl;
        return false;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception in train: " << e.what() << std::endl;
        return false;
    }
}

//����ͼ����з���
cv::Mat BayesianClassifier::classify(const cv::Mat& image) {
    try {
        //���ö�������״̬
        hasExtraFeature = false;
        extraFeatureImage = cv::Mat();

        return classifyWithExtraFeature(image, cv::Mat());
    }
    catch (const cv::Exception& e) {
        std::cerr << "OpenCV exception in classify: " << e.what() << std::endl;
        return cv::Mat();
    }
    catch (const std::exception& e) {
        std::cerr << "Exception in classify: " << e.what() << std::endl;
        return cv::Mat();
    }
}

//�Դ��ж�������ͨ����ͼ����з���
cv::Mat BayesianClassifier::classify(const cv::Mat& image, const cv::Mat& extraFeature) {
    return classifyWithExtraFeature(image, extraFeature);
}

cv::Mat BayesianClassifier::classifyWithExtraFeature(const cv::Mat& image, const cv::Mat& extraFeature) {
    try {
        if (image.empty()) {
            std::cerr << "Empty input image" << std::endl;
            return cv::Mat();
        }

        if (selectedFeatures.empty()) {
            std::cerr << "No features selected, call train() first" << std::endl;
            return cv::Mat();
        }

        //���ö�������������ṩ��
        if (!extraFeature.empty()) {
            setExtraFeatureImage(extraFeature);
        }

        cv::Mat featureImage = extractFeatures(image);
        if (featureImage.empty()) {
            std::cerr << "Failed to extract features" << std::endl;
            return cv::Mat();
        }

        std::vector<cv::Mat> featureChannels;
        cv::split(featureImage, featureChannels);

        cv::Mat result = cv::Mat::zeros(image.size(), CV_8U);

        //��ÿ�����ؼ���������
        for (int i = 0; i < image.rows; i++) {
            for (int j = 0; j < image.cols; j++) {
                double logProbForeground = log(std::max(priorProbs[1], 1e-10));
                double logProbBackground = log(std::max(priorProbs[0], 1e-10));

                for (int f : selectedFeatures) {
                    if (f >= 0 && f < featureChannels.size()) {
                        float pixelValue = featureChannels[f].at<float>(i, j);

                        //���ֵ�Ƿ���Ч
                        if (!std::isfinite(pixelValue)) {
                            continue;
                        }

                        //�����˹�����ܶȺ���
                        double foreStdDev = std::max(stdDevs[1].at<float>(0, f), 1e-6f);
                        double backStdDev = std::max(stdDevs[0].at<float>(0, f), 1e-6f);

                        double foreProb = -0.5 * pow((pixelValue - means[1].at<float>(0, f)) / foreStdDev, 2)
                            - log(foreStdDev) - 0.5 * log(2 * M_PI);
                        double backProb = -0.5 * pow((pixelValue - means[0].at<float>(0, f)) / backStdDev, 2)
                            - log(backStdDev) - 0.5 * log(2 * M_PI);

                        //��ֹ��ֵ����
                        if (std::isfinite(foreProb)) {
                            logProbForeground += foreProb;
                        }
                        if (std::isfinite(backProb)) {
                            logProbBackground += backProb;
                        }
                    }
                }

                //�ȽϺ������
                if (std::isfinite(logProbForeground) && std::isfinite(logProbBackground)) {
                    if (logProbForeground > logProbBackground) {
                        result.at<uchar>(i, j) = 255;
                    }
                }
            }
        }

        return result;
    }
    catch (const cv::Exception& e) {
        std::cerr << "OpenCV exception in classifyWithExtraFeature: " << e.what() << std::endl;
        return cv::Mat();
    }
    catch (const std::exception& e) {
        std::cerr << "Exception in classifyWithExtraFeature: " << e.what() << std::endl;
        return cv::Mat();
    }
}

//��������ָ��
void BayesianClassifier::calculateMetrics(const cv::Mat& result, const cv::Mat& groundTruth,
    double& accuracy, double& precision, double& recall, double& f1) {
    int tp = 0, fp = 0, tn = 0, fn = 0;

    for (int i = 0; i < result.rows; i++) {
        for (int j = 0; j < result.cols; j++) {
            bool resFg = result.at<uchar>(i, j) > 0;
            bool gtFg = groundTruth.at<uchar>(i, j) > 0;

            if (resFg && gtFg) tp++;
            else if (resFg && !gtFg) fp++;
            else if (!resFg && !gtFg) tn++;
            else if (!resFg && gtFg) fn++;
        }
    }

    accuracy = static_cast<double>(tp + tn) / (tp + tn + fp + fn);
    precision = (tp + fp > 0) ? static_cast<double>(tp) / (tp + fp) : 0.0;
    recall = (tp + fn > 0) ? static_cast<double>(tp) / (tp + fn) : 0.0;
    f1 = (precision + recall > 0) ? 2 * precision * recall / (precision + recall) : 0.0;
}

//���ɻ���������ӻ�
cv::Mat BayesianClassifier::visualizeConfusionMatrix(const cv::Mat& result, const cv::Mat& groundTruth) {
    cv::Mat confusionViz = cv::Mat::zeros(groundTruth.size(), CV_8UC3);

    for (int i = 0; i < result.rows; i++) {
        for (int j = 0; j < result.cols; j++) {
            bool resFg = result.at<uchar>(i, j) > 0;
            bool gtFg = groundTruth.at<uchar>(i, j) > 0;

            if (resFg && gtFg) {
                confusionViz.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 255, 0);
            }
            else if (resFg && !gtFg) {
                confusionViz.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 255);
            }
            else if (!resFg && gtFg) {
                confusionViz.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 0, 0);
            }
            else {
                confusionViz.at<cv::Vec3b>(i, j) = cv::Vec3b(128, 128, 128);
            }
        }
    }

    return confusionViz;
}

//��Ϣ����ͼ
cv::Mat visualizeInformationGain(const std::vector<double>& gains, const std::vector<int>& selectedFeatures) {
    if (gains.empty()) return cv::Mat();

    int margin = 50;
    int barWidth = 20;
    int spacing = 5;
    int height = 400;
    int width = gains.size() * (barWidth + spacing) + 2 * margin;

    cv::Mat viz(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    double maxGain = *std::max_element(gains.begin(), gains.end());
    if (maxGain <= 0) maxGain = 1.0;

    cv::line(viz, cv::Point(margin, margin), cv::Point(margin, height - margin), cv::Scalar(0, 0, 0), 2);
    cv::line(viz, cv::Point(margin, height - margin),
        cv::Point(width - margin, height - margin), cv::Scalar(0, 0, 0), 2);

    for (int i = 0; i <= 10; i++) {
        int y = height - margin - (height - 2 * margin) * i / 10;
        cv::line(viz, cv::Point(margin - 5, y), cv::Point(margin, y), cv::Scalar(0, 0, 0), 1);

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1) << maxGain * i / 10.0;
        cv::putText(viz, oss.str(), cv::Point(margin - 45, y + 5),
            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
    }

    std::set<int> selectedSet(selectedFeatures.begin(), selectedFeatures.end());

    for (int i = 0; i < gains.size(); i++) {
        int x = margin + i * (barWidth + spacing);
        int barHeight = static_cast<int>((height - 2 * margin) * (gains[i] / maxGain));

        cv::Scalar color = selectedSet.find(i) != selectedSet.end() ?
            cv::Scalar(0, 200, 0) : cv::Scalar(200, 200, 200);

        cv::rectangle(viz,
            cv::Point(x, height - margin - barHeight),
            cv::Point(x + barWidth, height - margin),
            color, -1);

        if (i % 5 == 0) { 
            cv::putText(viz, std::to_string(i),
                cv::Point(x, height - margin + 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
        }
    }

    //���ӱ���ͱ�ǩ
    cv::putText(viz, "Feature Information Gain",
        cv::Point(width / 2 - 100, 30),
        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);

    cv::putText(viz, "Feature Index",
        cv::Point(width / 2 - 50, height - 10),
        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);

    cv::Point textOrg(15, height / 2 + 50);
    cv::putText(viz, "Information Gain", textOrg,
        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);

    return viz;
}

//�Ż��ı�Ҷ˹����
cv::Mat optimizedBayesClassify(const cv::Mat& image, const cv::Mat& mask) {
    BayesianClassifier classifier;
    classifier.setBasicBGROnly(false);
    classifier.setUseFeatureSelection(true);
    classifier.setAdvancedFeatures(true);
    classifier.setUseGaussianSmoothing(true); 

    if (!classifier.train(image, mask, 8)) {
        std::cerr << "�Ż���Ҷ˹������ѵ��ʧ��!" << std::endl;
        return cv::Mat();
    }

    return classifier.classify(image);
}

//��Ҷ˹+Fisher��Ϸ�����
cv::Mat bayesFisherHybridClassify(const cv::Mat& image, const cv::Mat& mask) {
    try {
        cv::Mat grayImage;
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

        //׼������
        std::vector<cv::Mat> channels;
        cv::split(image, channels);

        cv::Mat foregroundMask = mask > 0;
        cv::Mat backgroundMask = mask == 0;

        //����ÿ��ͨ���ľ�ֵ����
        std::vector<cv::Scalar> foregroundMeans(3);
        std::vector<cv::Scalar> backgroundMeans(3);

        for (int c = 0; c < 3; c++) {
            cv::Scalar fgMean, fgStddev, bgMean, bgStddev;
            cv::meanStdDev(channels[c], fgMean, fgStddev, foregroundMask);
            cv::meanStdDev(channels[c], bgMean, bgStddev, backgroundMask);

            foregroundMeans[c] = fgMean;
            backgroundMeans[c] = bgMean;
        }

        //�������ɢ�Ⱦ���
        cv::Mat betweenClass = cv::Mat::zeros(3, 3, CV_32F);
        cv::Mat meanDiff = cv::Mat(3, 1, CV_32F);

        for (int i = 0; i < 3; i++) {
            meanDiff.at<float>(i, 0) = foregroundMeans[i][0] - backgroundMeans[i][0];
        }

        betweenClass = meanDiff * meanDiff.t();

        //����ɢ�Ⱦ���
        cv::Mat withinClass = cv::Mat::zeros(3, 3, CV_32F);

        //�ԽǾ���
        for (int i = 0; i < 3; i++) {
            withinClass.at<float>(i, i) = 1.0;
        }

        cv::Mat eigenvalues, eigenvectors;
        cv::eigen(withinClass.inv() * betweenClass, eigenvalues, eigenvectors);

        cv::Mat projectionVector = eigenvectors.row(0).t();
        cv::Mat projectedChannel = cv::Mat::zeros(image.size(), CV_32F);

        for (int i = 0; i < image.rows; i++) {
            for (int j = 0; j < image.cols; j++) {
                float projection = 0.0f;
                for (int c = 0; c < 3; c++) {
                    projection += channels[c].at<uchar>(i, j) * projectionVector.at<float>(c, 0);
                }
                projectedChannel.at<float>(i, j) = projection;
            }
        }

        //ʹ��FisherͶӰ
        cv::Mat normalizedProjection;
        cv::normalize(projectedChannel, normalizedProjection, 0, 255, cv::NORM_MINMAX);
        normalizedProjection.convertTo(normalizedProjection, CV_8U);

        //ʹ����ǿ�ı�Ҷ˹������
        BayesianClassifier bayesClassifier;
        bayesClassifier.setBasicBGROnly(false);
        bayesClassifier.setUseFeatureSelection(true);
        bayesClassifier.setAdvancedFeatures(true);
        bayesClassifier.setExtraFeatureImage(normalizedProjection);

        if (!bayesClassifier.train(image, mask)) {
            std::cerr << "��Ҷ˹������ѵ��ʧ��!" << std::endl;
            return cv::Mat();
        }

        cv::Mat result = bayesClassifier.classify(image, normalizedProjection);
        return result;
    }
    catch (const cv::Exception& e) {
        std::cerr << "OpenCV exception in bayesFisherHybrid: " << e.what() << std::endl;
        return cv::Mat();
    }
    catch (const std::exception& e) {
        std::cerr << "Exception in bayesFisherHybrid: " << e.what() << std::endl;
        return cv::Mat();
    }
}
