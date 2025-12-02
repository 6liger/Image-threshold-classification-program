QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    bayesian_classifier.cpp \
    fisher_classifier.cpp \
    imageprocessor.cpp \
    main.cpp \
    mainwindow.cpp \
    widget.cpp

HEADERS += \
    bayesian_classifier.h \
    fisher_classifier.h \
    imageprocessor.h \
    mainwindow.h \
    widget.h

FORMS += \
    widget.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

win32: LIBS += -L$$PWD/../../../../opencv/opencv/build/x64/vc16/lib/ -lopencv_world4120


INCLUDEPATH += $$PWD/../../../../opencv/opencv/build/include
DEPENDPATH += $$PWD/../../../../opencv/opencv/build/include
INCLUDEPATH += $$PWD/../../../../opencv/opencv/build/include
DEPENDPATH += $$PWD/../../../../opencv/opencv/build/include

