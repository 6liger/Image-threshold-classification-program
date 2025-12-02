#include "mainwindow.h"
#include <QApplication>
#include <QDir>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , processor(new ImageProcessor(this))
{
    setupUI();

    connect(processor, &ImageProcessor::processingFinished,
            this, &MainWindow::onProcessingFinished);
    connect(processor, &ImageProcessor::progressUpdated,
            progressBar, &QProgressBar::setValue);
}

MainWindow::~MainWindow()
{
}

void MainWindow::setupUI()
{
    setWindowTitle("图像阈值分割工具");
    setMinimumSize(1200, 800);

    centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);

    mainSplitter = new QSplitter(Qt::Horizontal, this);

    // 左侧控制面板
    controlPanel = new QWidget();
    controlPanel->setMaximumWidth(350);
    controlPanel->setMinimumWidth(300);

    QVBoxLayout *controlLayout = new QVBoxLayout(controlPanel);

    // 文件操作按钮
    openButton = new QPushButton("打开图像");
    processButton = new QPushButton("开始处理");
    saveButton = new QPushButton("保存结果");

    processButton->setEnabled(false);
    saveButton->setEnabled(false);

    connect(openButton, &QPushButton::clicked, this, &MainWindow::openImage);
    connect(processButton, &QPushButton::clicked, this, &MainWindow::processImage);
    connect(saveButton, &QPushButton::clicked, this, &MainWindow::saveResults);

    // 参数控制组
    parameterGroup = new QGroupBox("处理参数");
    QGridLayout *paramLayout = new QGridLayout(parameterGroup);

    paramLayout->addWidget(new QLabel("手动阈值:"), 0, 0);
    manualThresholdSpin = new QSpinBox();
    manualThresholdSpin->setRange(0, 255);
    manualThresholdSpin->setValue(170);
    paramLayout->addWidget(manualThresholdSpin, 0, 1);

    paramLayout->addWidget(new QLabel("高斯核大小:"), 1, 0);
    gaussianKernelSpin = new QSpinBox();
    gaussianKernelSpin->setRange(3, 31);
    gaussianKernelSpin->setValue(5);
    gaussianKernelSpin->setSingleStep(2); // 只允许奇数
    paramLayout->addWidget(gaussianKernelSpin, 1, 1);

    paramLayout->addWidget(new QLabel("高斯σ值:"), 2, 0);
    gaussianSigmaSpin = new QDoubleSpinBox();
    gaussianSigmaSpin->setRange(0.1, 10.0);
    gaussianSigmaSpin->setValue(1.5);
    gaussianSigmaSpin->setSingleStep(0.1);
    paramLayout->addWidget(gaussianSigmaSpin, 2, 1);

    paramLayout->addWidget(new QLabel("中值核大小:"), 3, 0);
    medianKernelSpin = new QSpinBox();
    medianKernelSpin->setRange(3, 31);
    medianKernelSpin->setValue(5);
    medianKernelSpin->setSingleStep(2);
    paramLayout->addWidget(medianKernelSpin, 3, 1);

    // 处理方法选择组
    methodGroup = new QGroupBox("处理方法");
    QVBoxLayout *methodLayout = new QVBoxLayout(methodGroup);

    singleChannelCheck = new QCheckBox("单通道处理");
    dualChannelCheck = new QCheckBox("双通道处理");
    filteringCheck = new QCheckBox("滤波处理");
    illuminationCheck = new QCheckBox("光照校正");

    singleChannelCheck->setChecked(true);
    dualChannelCheck->setChecked(true);
    filteringCheck->setChecked(true);
    illuminationCheck->setChecked(true);

    methodLayout->addWidget(singleChannelCheck);
    methodLayout->addWidget(dualChannelCheck);
    methodLayout->addWidget(filteringCheck);
    methodLayout->addWidget(illuminationCheck);

    // 分类方法
    classifierGroup = new QGroupBox("统计分类");
    QGridLayout *classifierLayout = new QGridLayout(classifierGroup);

    bayesCheck = new QCheckBox("启用贝叶斯分类");
    bayesBasicCheck = new QCheckBox("仅使用BGR通道");
    bayesAdvancedCheck = new QCheckBox("启用高级特征");
    bayesFeatureSelectionCheck = new QCheckBox("信息增益选特征");
    bayesGaussianCheck = new QCheckBox("预先高斯平滑");
    bayesFeatureSpin = new QSpinBox();
    bayesFeatureSpin->setRange(1, 32);
    bayesFeatureSpin->setValue(8);

    bayesAdvancedCheck->setChecked(true);
    bayesFeatureSelectionCheck->setChecked(true);
    bayesGaussianCheck->setChecked(true);

    classifierLayout->addWidget(bayesCheck, 0, 0, 1, 2);
    classifierLayout->addWidget(bayesBasicCheck, 1, 0, 1, 2);
    classifierLayout->addWidget(bayesAdvancedCheck, 2, 0, 1, 2);
    classifierLayout->addWidget(bayesFeatureSelectionCheck, 3, 0, 1, 2);
    classifierLayout->addWidget(bayesGaussianCheck, 4, 0, 1, 2);
    classifierLayout->addWidget(new QLabel("特征数量:"), 5, 0);
    classifierLayout->addWidget(bayesFeatureSpin, 5, 1);

    connect(bayesFeatureSelectionCheck, &QCheckBox::toggled,
            bayesFeatureSpin, &QWidget::setEnabled);
    bayesFeatureSpin->setEnabled(bayesFeatureSelectionCheck->isChecked());

    fisherSingleCheck = new QCheckBox("Fisher 单通道");
    fisherMultiCheck = new QCheckBox("Fisher 多通道");
    fisherChannelCombo = new QComboBox();
    fisherChannelCombo->addItems({"蓝通道", "绿通道", "红通道"});
    fisherChannelCombo->setCurrentIndex(2);
    fisherMorphCheck = new QCheckBox("形态学优化");
    fisherMorphCheck->setChecked(true);
    fisherMorphSpin = new QSpinBox();
    fisherMorphSpin->setRange(3, 21);
    fisherMorphSpin->setSingleStep(2);
    fisherMorphSpin->setValue(5);

    classifierLayout->addWidget(fisherSingleCheck, 6, 0, 1, 2);
    classifierLayout->addWidget(new QLabel("单通道选择:"), 7, 0);
    classifierLayout->addWidget(fisherChannelCombo, 7, 1);
    classifierLayout->addWidget(fisherMultiCheck, 8, 0, 1, 2);
    classifierLayout->addWidget(fisherMorphCheck, 9, 0, 1, 2);
    classifierLayout->addWidget(new QLabel("形态学核大小:"), 10, 0);
    classifierLayout->addWidget(fisherMorphSpin, 10, 1);

    connect(fisherMorphCheck, &QCheckBox::toggled,
            fisherMorphSpin, &QWidget::setEnabled);
    fisherMorphSpin->setEnabled(fisherMorphCheck->isChecked());

    // 进度条
    progressBar = new QProgressBar();

    // 日志区域
    logTextEdit = new QTextEdit();
    logTextEdit->setMaximumHeight(200);
    logTextEdit->setPlainText("等待打开图像...");

    // 布局
    controlLayout->addWidget(openButton);
    controlLayout->addWidget(processButton);
    controlLayout->addWidget(saveButton);
    controlLayout->addWidget(parameterGroup);
    controlLayout->addWidget(methodGroup);
    controlLayout->addWidget(classifierGroup);
    controlLayout->addWidget(progressBar);
    controlLayout->addWidget(new QLabel("处理日志:"));
    controlLayout->addWidget(logTextEdit);
    controlLayout->addStretch();

    // 右侧结果显示区域
    resultWidget = new QWidget();
    resultGridLayout = new QGridLayout(resultWidget);
    resultScrollArea = new QScrollArea();
    resultScrollArea->setWidget(resultWidget);
    resultScrollArea->setWidgetResizable(true);

    // 添加到主分割器
    mainSplitter->addWidget(controlPanel);
    mainSplitter->addWidget(resultScrollArea);
    mainSplitter->setStretchFactor(0, 0);
    mainSplitter->setStretchFactor(1, 1);

    QHBoxLayout *mainLayout = new QHBoxLayout(centralWidget);
    mainLayout->addWidget(mainSplitter);
}

void MainWindow::openImage()
{
    QString fileName = QFileDialog::getOpenFileName(
        this, "选择图像文件", "",
        "图像文件 (*.png *.jpg *.jpeg *.bmp *.tiff)");

    if (!fileName.isEmpty()) {
        currentImagePath = fileName;
        processor->setImage(fileName);
        processButton->setEnabled(true);
        logTextEdit->append(QString("已加载图像: %1").arg(QFileInfo(fileName).fileName()));
        clearResults();
    }
}

void MainWindow::processImage()
{
    if (currentImagePath.isEmpty()) {
        QMessageBox::warning(this, "警告", "请先选择一张图像！");
        return;
    }

    // 设置处理参数
    ProcessingParameters params;
    params.manualThreshold = manualThresholdSpin->value();
    params.gaussianKernelSize = gaussianKernelSpin->value();
    params.gaussianSigma = gaussianSigmaSpin->value();
    params.medianKernelSize = medianKernelSpin->value();
    params.enableSingleChannel = singleChannelCheck->isChecked();
    params.enableDualChannel = dualChannelCheck->isChecked();
    params.enableFiltering = filteringCheck->isChecked();
    params.enableIllumination = illuminationCheck->isChecked();
    params.enableBayesian = bayesCheck->isChecked();
    params.bayesUseBasicBGR = bayesBasicCheck->isChecked();
    params.bayesUseAdvancedFeatures = bayesAdvancedCheck->isChecked();
    params.bayesUseFeatureSelection = bayesFeatureSelectionCheck->isChecked();
    params.bayesUseGaussianSmoothing = bayesGaussianCheck->isChecked();
    params.bayesFeatureCount = bayesFeatureSpin->value();
    params.enableFisherSingle = fisherSingleCheck->isChecked();
    params.enableFisherMulti = fisherMultiCheck->isChecked();
    params.fisherSingleChannel = fisherChannelCombo->currentIndex();
    params.fisherUseMorphology = fisherMorphCheck->isChecked();
    params.fisherMorphSize = fisherMorphSpin->value();

    processor->setParameters(params);

    processButton->setEnabled(false);
    logTextEdit->append("开始处理图像...");

    // 在后台线程中处理
    QThread::create([this]() {
        processor->processImage();
    })->start();
}

void MainWindow::onProcessingFinished(const QString& results)
{
    processButton->setEnabled(true);
    saveButton->setEnabled(true);
    logTextEdit->append("处理完成！");
    logTextEdit->append(results);

    updateImageDisplay();
}

void MainWindow::updateImageDisplay()
{
    clearResults();

    auto results = processor->getResults();
    int cols = 3; // 每行显示3张图

    for (int i = 0; i < results.size(); ++i) {
        QLabel *label = new QLabel();
        label->setPixmap(results[i].second.scaled(300, 300, Qt::KeepAspectRatio, Qt::SmoothTransformation));
        label->setAlignment(Qt::AlignCenter);
        label->setFrameStyle(QFrame::Box);
        label->setToolTip(results[i].first);

        QLabel *titleLabel = new QLabel(results[i].first);
        titleLabel->setAlignment(Qt::AlignCenter);
        titleLabel->setWordWrap(true);

        QVBoxLayout *itemLayout = new QVBoxLayout();
        itemLayout->addWidget(label);
        itemLayout->addWidget(titleLabel);

        QWidget *itemWidget = new QWidget();
        itemWidget->setLayout(itemLayout);

        int row = i / cols;
        int col = i % cols;
        resultGridLayout->addWidget(itemWidget, row, col);

        resultLabels.append(label);
    }
}

void MainWindow::clearResults()
{
    for (auto label : resultLabels) {
        label->deleteLater();
    }
    resultLabels.clear();

    // 清除网格布局中的所有项目
    QLayoutItem *item;
    while ((item = resultGridLayout->takeAt(0)) != nullptr) {
        delete item->widget();
        delete item;
    }
}

void MainWindow::saveResults()
{
    QString dirPath = QFileDialog::getExistingDirectory(
        this, "选择保存目录", "");

    if (!dirPath.isEmpty()) {
        auto results = processor->getResults();
        int saved = 0;

        for (const auto& result : results) {
            // 修改这一行，明确指定字符类型
            QString cleanName = result.first;
            cleanName.replace(' ', '_');
            cleanName.replace('/', '_');
            cleanName.replace('\\', '_'); // 也处理反斜杠
            cleanName.replace(':', '_');  // 处理冒号（Windows文件名不允许）

            QString fileName = QString("%1/%2.png")
                                   .arg(dirPath)
                                   .arg(cleanName);

            if (result.second.save(fileName)) {
                saved++;
            }
        }

        QMessageBox::information(this, "保存完成",
                                 QString("成功保存 %1/%2 张图像到:\n%3")
                                     .arg(saved).arg(results.size()).arg(dirPath));

        logTextEdit->append(QString("结果已保存到: %1").arg(dirPath));
    }
}

