#include "fisher_classifier.h"
#include <algorithm>
#include <limits>

void FisherClassifier::setEnableMorphology(bool enable) {
    enableMorphology = enable;
}

void FisherClassifier::setMorphologySize(int size) {
    morphologySize = size;
}

bool FisherClassifier::trainSingleBand(const cv::Mat& image, const cv::Mat& mask, int channel) {
    try {
        //转换为灰度图或提取单通道
        cv::Mat grayImage;
        if (image.channels() == 1) {
            grayImage = image.clone();
        }
        else if (image.channels() == 3 && channel >= 0 && channel < 3) {
            std::vector<cv::Mat> channels;
            cv::split(image, channels);
            grayImage = channels[channel];
        }
        else {
            std::cerr << "Invalid image format or channel selection for single-band Fisher" << std::endl;
            return false;
        }

        //转换为浮点类型
        cv::Mat floatImage;
        grayImage.convertTo(floatImage, CV_32F);

        //分离前景和背景
        cv::Mat foregroundMask = mask > 0;
        cv::Mat backgroundMask = mask == 0;

        std::vector<float> foregroundValues, backgroundValues;

        for (int i = 0; i < floatImage.rows; i++) {
            for (int j = 0; j < floatImage.cols; j++) {
                if (foregroundMask.at<uchar>(i, j)) {
                    foregroundValues.push_back(floatImage.at<float>(i, j));
                }
                else if (backgroundMask.at<uchar>(i, j)) {
                    backgroundValues.push_back(floatImage.at<float>(i, j));
                }
            }
        }

        //计算类均值
        float foregroundMean = 0, backgroundMean = 0;

        for (float val : foregroundValues) foregroundMean += val;
        for (float val : backgroundValues) backgroundMean += val;

        if (foregroundValues.empty() || backgroundValues.empty()) {
            std::cerr << "Missing foreground or background pixels" << std::endl;
            return false;
        }

        foregroundMean /= foregroundValues.size();
        backgroundMean /= backgroundValues.size();

        //计算类内方差
        float foregroundVar = 0, backgroundVar = 0;

        for (float val : foregroundValues) {
            float diff = val - foregroundMean;
            foregroundVar += diff * diff;
        }

        for (float val : backgroundValues) {
            float diff = val - backgroundMean;
            backgroundVar += diff * diff;
        }

        foregroundVar /= foregroundValues.size();
        backgroundVar /= backgroundValues.size();

        //计算类内散布
        float withinClassScatter = foregroundVar + backgroundVar;

        //计算类间距离平方
        float betweenClassDistance = foregroundMean - backgroundMean;
        float betweenClassScatter = betweenClassDistance * betweenClassDistance;

        //计算Fisher比率
        this->fisherRatio = withinClassScatter > 0 ? betweenClassScatter / withinClassScatter : 0;

        //设置投影向量 
        projectionVector = cv::Mat(1, 1, CV_32F);
        projectionVector.at<float>(0, 0) = 1.0f; 

        //设置阈值
        threshold = (foregroundMean + backgroundMean) / 2.0f;

        std::cout << "Single-band Fisher analysis results:" << std::endl;
        std::cout << "  Foreground mean: " << foregroundMean << std::endl;
        std::cout << "  Background mean: " << backgroundMean << std::endl;
        std::cout << "  Fisher ratio: " << this->fisherRatio << std::endl;
        std::cout << "  Classification threshold: " << threshold << std::endl;

        isTrained = true;
        return true;
    }
    catch (const cv::Exception& e) {
        std::cerr << "OpenCV exception in trainSingleBand: " << e.what() << std::endl;
        return false;
    }
}

bool FisherClassifier::trainMultiBand(const cv::Mat& image, const cv::Mat& mask) {
    try {
        if (image.channels() != 3) {
            std::cerr << "Multi-band Fisher requires a 3-channel image" << std::endl;
            return false;
        }

        cv::Mat foregroundMask = mask > 0;
        cv::Mat backgroundMask = mask == 0;

        int numForegroundPixels = cv::countNonZero(foregroundMask);
        int numBackgroundPixels = cv::countNonZero(backgroundMask);

        if (numForegroundPixels == 0 || numBackgroundPixels == 0) {
            std::cerr << "Missing foreground or background pixels in mask" << std::endl;
            return false;
        }

        //将图像转换为浮点类型
        cv::Mat floatImage;
        image.convertTo(floatImage, CV_32F);

        //准备数据
        std::vector<cv::Mat> channels;
        cv::split(floatImage, channels);
        cv::Mat foregroundData(numForegroundPixels, 3, CV_32F);
        cv::Mat backgroundData(numBackgroundPixels, 3, CV_32F);

        int fgIdx = 0, bgIdx = 0;
        for (int i = 0; i < image.rows; i++) {
            for (int j = 0; j < image.cols; j++) {
                if (foregroundMask.at<uchar>(i, j)) {
                    for (int c = 0; c < 3; c++) {
                        foregroundData.at<float>(fgIdx, c) = channels[c].at<float>(i, j);
                    }
                    fgIdx++;
                }
                else if (backgroundMask.at<uchar>(i, j)) {
                    for (int c = 0; c < 3; c++) {
                        backgroundData.at<float>(bgIdx, c) = channels[c].at<float>(i, j);
                    }
                    bgIdx++;
                }
            }
        }

        //计算类均值向量
        cv::Mat foregroundMean = cv::Mat::zeros(1, 3, CV_32F);
        cv::Mat backgroundMean = cv::Mat::zeros(1, 3, CV_32F);

        for (int i = 0; i < 3; i++) {
            cv::Scalar meanFg = cv::mean(foregroundData.col(i));
            cv::Scalar meanBg = cv::mean(backgroundData.col(i));
            foregroundMean.at<float>(0, i) = meanFg[0];
            backgroundMean.at<float>(0, i) = meanBg[0];
        }

        //计算类内散度矩阵Sw和类间散度矩阵Sb
        cv::Mat Sw = cv::Mat::zeros(3, 3, CV_32F);

        //计算每个类的协方差矩阵
        cv::Mat covFg = cv::Mat::zeros(3, 3, CV_32F);
        cv::Mat covBg = cv::Mat::zeros(3, 3, CV_32F);

        //前景协方差
        for (int i = 0; i < numForegroundPixels; i++) {
            cv::Mat diff = foregroundData.row(i) - foregroundMean;
            covFg += diff.t() * diff;
        }
        covFg /= numForegroundPixels;

        //背景协方差
        for (int i = 0; i < numBackgroundPixels; i++) {
            cv::Mat diff = backgroundData.row(i) - backgroundMean;
            covBg += diff.t() * diff;
        }
        covBg /= numBackgroundPixels;

        //类内散度矩阵为两个类协方差的加权和
        Sw = (numForegroundPixels * covFg + numBackgroundPixels * covBg) /
            (numForegroundPixels + numBackgroundPixels);

        //类间散度矩阵
        cv::Mat meanDiff = foregroundMean - backgroundMean;
        cv::Mat Sb = meanDiff.t() * meanDiff;

        //计算Sw的逆
        cv::Mat SwInv;
        cv::invert(Sw, SwInv, cv::DECOMP_SVD);

        projectionVector = SwInv * meanDiff.t();
        double norm = cv::norm(projectionVector);
        if (norm > 0) {
            projectionVector /= norm;
        }

        //将两类数据投影到Fisher方向上
        cv::Mat projectedFg = foregroundData * projectionVector;
        cv::Mat projectedBg = backgroundData * projectionVector;

        //计算投影后的均值
        double projectedFgMean = cv::mean(projectedFg)[0];
        double projectedBgMean = cv::mean(projectedBg)[0];

        //计算阈值
        threshold = (projectedFgMean + projectedBgMean) / 2.0;

        //计算Fisher比率
        double betweenClassVar = pow(projectedFgMean - projectedBgMean, 2);
        cv::Scalar fgScatter, bgScatter;
        cv::meanStdDev(projectedFg, cv::Scalar(), fgScatter);
        cv::meanStdDev(projectedBg, cv::Scalar(), bgScatter);
        double withinClassVar = pow(fgScatter[0], 2) + pow(bgScatter[0], 2);
        fisherRatio = betweenClassVar / (withinClassVar > 0 ? withinClassVar : 1e-10);

        std::cout << "Multi-band Fisher analysis results:" << std::endl;
        std::cout << "  Projection vector: ["
            << projectionVector.at<float>(0, 0) << ", "
            << projectionVector.at<float>(1, 0) << ", "
            << projectionVector.at<float>(2, 0) << "]" << std::endl;
        std::cout << "  Projected foreground mean: " << projectedFgMean << std::endl;
        std::cout << "  Projected background mean: " << projectedBgMean << std::endl;
        std::cout << "  Fisher ratio: " << fisherRatio << std::endl;
        std::cout << "  Classification threshold: " << threshold << std::endl;

        isTrained = true;
        return true;
    }
    catch (const cv::Exception& e) {
        std::cerr << "OpenCV exception in trainMultiBand: " << e.what() << std::endl;
        return false;
    }
}

cv::Mat FisherClassifier::classifySingleBand(const cv::Mat& image, int channel) {
    try {
        if (!isTrained) {
            std::cerr << "Classifier not trained yet" << std::endl;
            return cv::Mat();
        }

        //提取目标通道
        cv::Mat targetChannel;
        if (image.channels() == 1) {
            targetChannel = image.clone();
        }
        else if (image.channels() == 3 && channel >= 0 && channel < 3) {
            std::vector<cv::Mat> channels;
            cv::split(image, channels);
            targetChannel = channels[channel];
        }
        else {
            std::cerr << "Invalid image format or channel selection" << std::endl;
            return cv::Mat();
        }

        cv::Mat floatImage;
        targetChannel.convertTo(floatImage, CV_32F);

        cv::Mat result = cv::Mat::zeros(image.size(), CV_8U);

        int totalPixels = image.rows * image.cols;
        int aboveThreshold = 0;

        for (int i = 0; i < floatImage.rows; i++) {
            for (int j = 0; j < floatImage.cols; j++) {
                if (floatImage.at<float>(i, j) > threshold) {
                    aboveThreshold++;
                }
            }
        }

        //假设前景像素较少
        bool invertThreshold = aboveThreshold > totalPixels / 2;

        for (int i = 0; i < floatImage.rows; i++) {
            for (int j = 0; j < floatImage.cols; j++) {
                if (invertThreshold) {
                    result.at<uchar>(i, j) = (floatImage.at<float>(i, j) < threshold) ? 255 : 0;
                }
                else {
                    result.at<uchar>(i, j) = (floatImage.at<float>(i, j) > threshold) ? 255 : 0;
                }
            }
        }

        //形态学后处理
        if (enableMorphology) {
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                cv::Size(morphologySize, morphologySize));
            //先开运算去噪点，再闭运算填充空洞
            cv::morphologyEx(result, result, cv::MORPH_OPEN, kernel);
            cv::morphologyEx(result, result, cv::MORPH_CLOSE, kernel);
        }

        return result;
    }
    catch (const cv::Exception& e) {
        std::cerr << "OpenCV exception in classifySingleBand: " << e.what() << std::endl;
        return cv::Mat();
    }
}

cv::Mat FisherClassifier::classifyMultiBand(const cv::Mat& image) {
    try {
        if (!isTrained || projectionVector.empty()) {
            std::cerr << "Classifier not properly trained yet" << std::endl;
            return cv::Mat();
        }

        if (image.channels() != 3) {
            std::cerr << "Multi-band Fisher requires a 3-channel image" << std::endl;
            return cv::Mat();
        }

        cv::Mat floatImage;
        image.convertTo(floatImage, CV_32F);
        std::vector<cv::Mat> channels;
        cv::split(floatImage, channels);

        cv::Mat result = cv::Mat::zeros(image.size(), CV_8U);

        //计算所有像素的投影值
        cv::Mat projectionValues = cv::Mat::zeros(image.rows, image.cols, CV_32F);
        int aboveThreshold = 0;
        int totalPixels = image.rows * image.cols;

        for (int i = 0; i < image.rows; i++) {
            for (int j = 0; j < image.cols; j++) {
                float projValue = 0;
                for (int c = 0; c < 3; c++) {
                    projValue += channels[c].at<float>(i, j) * projectionVector.at<float>(c, 0);
                }
                projectionValues.at<float>(i, j) = projValue;

                if (projValue > threshold) {
                    aboveThreshold++;
                }
            }
        }

        //自适应判断前景/背景
        bool invertThreshold = aboveThreshold > totalPixels / 2;

        for (int i = 0; i < image.rows; i++) {
            for (int j = 0; j < image.cols; j++) {
                float projValue = projectionValues.at<float>(i, j);

                if (invertThreshold) {
                    //多数像素高于阈值，低值为前景
                    result.at<uchar>(i, j) = (projValue < threshold) ? 255 : 0;
                }
                else {
                    //多数像素低于阈值，高值为前景
                    result.at<uchar>(i, j) = (projValue > threshold) ? 255 : 0;
                }
            }
        }

        //形态学后处理
        if (enableMorphology) {
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                cv::Size(morphologySize, morphologySize));
            //先开运算去噪点，再闭运算填充空洞
            cv::morphologyEx(result, result, cv::MORPH_OPEN, kernel);
            cv::morphologyEx(result, result, cv::MORPH_CLOSE, kernel);
        }

        return result;
    }
    catch (const cv::Exception& e) {
        std::cerr << "OpenCV exception in classifyMultiBand: " << e.what() << std::endl;
        return cv::Mat();
    }
}

double FisherClassifier::getFisherRatio() const {
    return fisherRatio;
}

cv::Mat FisherClassifier::visualizeProjectionDistribution(const cv::Mat& image, const cv::Mat& mask) {
    try {
        if (!isTrained || projectionVector.empty()) {
            std::cerr << "Classifier not properly trained" << std::endl;
            return cv::Mat();
        }

        //将图像投影到Fisher方向
        cv::Mat floatImage;
        image.convertTo(floatImage, CV_32F);

        cv::Mat foregroundMask = mask > 0;
        cv::Mat backgroundMask = mask == 0;

        std::vector<float> foregroundProjections, backgroundProjections;

        //对于三波段图像
        if (image.channels() == 3 && projectionVector.rows == 3) {
            std::vector<cv::Mat> channels;
            cv::split(floatImage, channels);

            for (int i = 0; i < image.rows; i++) {
                for (int j = 0; j < image.cols; j++) {
                    float projValue = 0;
                    for (int c = 0; c < 3; c++) {
                        projValue += channels[c].at<float>(i, j) * projectionVector.at<float>(c, 0);
                    }

                    if (foregroundMask.at<uchar>(i, j)) {
                        foregroundProjections.push_back(projValue);
                    }
                    else if (backgroundMask.at<uchar>(i, j)) {
                        backgroundProjections.push_back(projValue);
                    }
                }
            }
        }
        //对于单波段图像
        else if (image.channels() == 1 || (image.channels() == 3 && projectionVector.rows == 1)) {
            cv::Mat grayImage;
            if (image.channels() == 1) {
                grayImage = floatImage.clone();
            }
            else {
                cv::cvtColor(floatImage, grayImage, cv::COLOR_BGR2GRAY);
            }

            for (int i = 0; i < grayImage.rows; i++) {
                for (int j = 0; j < grayImage.cols; j++) {
                    if (foregroundMask.at<uchar>(i, j)) {
                        foregroundProjections.push_back(grayImage.at<float>(i, j));
                    }
                    else if (backgroundMask.at<uchar>(i, j)) {
                        backgroundProjections.push_back(grayImage.at<float>(i, j));
                    }
                }
            }
        }
        else {
            std::cerr << "Incompatible image and projection vector" << std::endl;
            return cv::Mat();
        }

        //查找投影值的最小和最大值
        float minProj = std::numeric_limits<float>::max();
        float maxProj = std::numeric_limits<float>::lowest();

        for (float p : foregroundProjections) {
            minProj = std::min(minProj, p);
            maxProj = std::max(maxProj, p);
        }

        for (float p : backgroundProjections) {
            minProj = std::min(minProj, p);
            maxProj = std::max(maxProj, p);
        }

        //为了可视化，扩展范围一点
        float range = maxProj - minProj;
        minProj -= range * 0.1f;
        maxProj += range * 0.1f;

        //创建直方图
        const int histSize = 100;
        float binWidth = (maxProj - minProj) / histSize;

        std::vector<int> fgHist(histSize, 0), bgHist(histSize, 0);

        //填充直方图
        for (float p : foregroundProjections) {
            int bin = std::min(histSize - 1, std::max(0, int((p - minProj) / binWidth)));
            fgHist[bin]++;
        }

        for (float p : backgroundProjections) {
            int bin = std::min(histSize - 1, std::max(0, int((p - minProj) / binWidth)));
            bgHist[bin]++;
        }

        //找到直方图的最大值，用于归一化
        int maxCount = 0;
        for (int i = 0; i < histSize; i++) {
            maxCount = std::max(maxCount, std::max(fgHist[i], bgHist[i]));
        }

        //创建可视化图像
        const int histHeight = 300;
        const int histWidth = 600;
        const int margin = 50;

        cv::Mat histImage(histHeight + 2 * margin, histWidth + 2 * margin, CV_8UC3, cv::Scalar(255, 255, 255));

        //绘制坐标轴
        cv::line(histImage,
            cv::Point(margin, histHeight + margin),
            cv::Point(histWidth + margin, histHeight + margin),
            cv::Scalar(0, 0, 0), 2);

        cv::line(histImage,
            cv::Point(margin, margin),
            cv::Point(margin, histHeight + margin),
            cv::Scalar(0, 0, 0), 2);

        int thresholdBin = int((threshold - minProj) / binWidth);
        thresholdBin = std::min(histSize - 1, std::max(0, thresholdBin));
        int thresholdX = margin + thresholdBin * histWidth / histSize;

        cv::line(histImage,
            cv::Point(thresholdX, margin),
            cv::Point(thresholdX, histHeight + margin),
            cv::Scalar(0, 0, 255), 2);
        for (int i = 0; i < histSize; i++) {
            int x = margin + i * histWidth / histSize;

            int fgHeight = histHeight * fgHist[i] / maxCount;
            if (fgHeight > 0) {
                cv::rectangle(histImage,
                    cv::Point(x, histHeight + margin - fgHeight),
                    cv::Point(x + histWidth / histSize - 1, histHeight + margin),
                    cv::Scalar(0, 200, 0),
                    -1);
            }

            int bgHeight = histHeight * bgHist[i] / maxCount;
            if (bgHeight > 0) {
                cv::Mat roi = histImage(cv::Rect(x, histHeight + margin - bgHeight,
                    histWidth / histSize - 1, bgHeight));
                cv::Mat overlay = roi.clone();
                cv::rectangle(overlay, cv::Rect(0, 0, overlay.cols, overlay.rows),
                    cv::Scalar(200, 0, 0), -1);
                cv::addWeighted(overlay, 0.5, roi, 0.5, 0, roi);
            }
        }

        //添加标签和文本
        cv::putText(histImage, "Fisher Projection Distribution",
            cv::Point(histWidth / 2, margin / 2),
            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);

        cv::putText(histImage, "Projection Value",
            cv::Point(histWidth / 2, histHeight + margin + margin / 2),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);

        cv::putText(histImage, "Frequency",
            cv::Point(margin / 4, histHeight / 2),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);

        //添加图例
        cv::rectangle(histImage, cv::Point(histWidth, margin),
            cv::Point(histWidth + 20, margin + 20),
            cv::Scalar(0, 200, 0), -1);
        cv::putText(histImage, "Foreground", cv::Point(histWidth + 30, margin + 15),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

        cv::rectangle(histImage, cv::Point(histWidth, margin + 30),
            cv::Point(histWidth + 20, margin + 50),
            cv::Scalar(200, 0, 0), -1);
        cv::putText(histImage, "Background", cv::Point(histWidth + 30, margin + 45),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

        cv::putText(histImage, "Threshold", cv::Point(thresholdX - 40, margin - 10),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);

        return histImage;
    }
    catch (const cv::Exception& e) {
        std::cerr << "OpenCV exception in visualizeProjectionDistribution: " << e.what() << std::endl;
        return cv::Mat();
    }
}
