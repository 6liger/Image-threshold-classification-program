#include <QApplication>
#include "mainwindow.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    app.setApplicationName("图像阈值分割工具");
    app.setApplicationVersion("1.0");

    MainWindow window;
    window.show();

    return app.exec();
}
