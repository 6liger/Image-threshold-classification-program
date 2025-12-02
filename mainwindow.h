#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QtWidgets/QMainWindow>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QLabel>
#include <QtWidgets/QScrollArea>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QProgressBar>
#include <QtWidgets/QTextEdit>
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QSplitter>

#include "imageprocessor.h"

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void openImage();
    void processImage();
    void saveResults();
    void onProcessingFinished(const QString& results);

private:
    void setupUI();
    void updateImageDisplay();
    void clearResults();

    // UI组件
    QWidget *centralWidget;
    QSplitter *mainSplitter;

    // 左侧控制面板
    QWidget *controlPanel;
    QPushButton *openButton;
    QPushButton *processButton;
    QPushButton *saveButton;

    // 参数控制组
    QGroupBox *parameterGroup;
    QSpinBox *manualThresholdSpin;
    QSpinBox *gaussianKernelSpin;
    QDoubleSpinBox *gaussianSigmaSpin;
    QSpinBox *medianKernelSpin;
    QGroupBox *classifierGroup;

    // 处理选项组
    QGroupBox *methodGroup;
    QCheckBox *singleChannelCheck;
    QCheckBox *dualChannelCheck;
    QCheckBox *filteringCheck;
    QCheckBox *illuminationCheck;
    QCheckBox *bayesCheck;
    QCheckBox *bayesBasicCheck;
    QCheckBox *bayesAdvancedCheck;
    QCheckBox *bayesFeatureSelectionCheck;
    QCheckBox *bayesGaussianCheck;
    QSpinBox *bayesFeatureSpin;
    QCheckBox *fisherSingleCheck;
    QCheckBox *fisherMultiCheck;
    QComboBox *fisherChannelCombo;
    QCheckBox *fisherMorphCheck;
    QSpinBox *fisherMorphSpin;

    // 进度条和日志
    QProgressBar *progressBar;
    QTextEdit *logTextEdit;

    // 右侧图像显示区域
    QScrollArea *imageScrollArea;
    QLabel *imageLabel;
    QGridLayout *resultGridLayout;
    QWidget *resultWidget;
    QScrollArea *resultScrollArea;

    // 数据
    ImageProcessor *processor;
    QString currentImagePath;
    QList<QLabel*> resultLabels;
};

#endif // MAINWINDOW_H
