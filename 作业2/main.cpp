#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <chrono>

// 定义装甲板类型
enum class ArmorType { SMALL, LARGE };

// 定义装甲板结构体
struct Armor {
    ArmorType type;
    // 其他属性...
    cv::Rect boundingRect;  // 添加矩形边界框成员
};

// 二值化函数
cv::Mat binarizeImage(const cv::Mat& src, int threshold) {
    cv::Mat binaryImage;
    cv::cvtColor(src, binaryImage, cv::COLOR_BGR2GRAY);
    cv::threshold(binaryImage, binaryImage, threshold, 255, cv::THRESH_BINARY);
    return binaryImage;
}

// 装甲板检测函数
std::vector<Armor> detectArmors(const cv::Mat& binaryImage, const cv::Mat& colorImage) {
    // 颜色过滤，假设红色为目标颜色
    cv::Mat filteredImage;
    std::vector<cv::Mat> channels;
    cv::split(colorImage, channels);  // 拆分通道
    cv::inRange(channels[2], 100, 255, filteredImage);  // 使用红色通道进行过滤

    // 合并二值化图像和颜色过滤图像
    cv::Mat combinedImage;
    cv::bitwise_and(binaryImage, filteredImage, combinedImage);

    // 膨胀和腐蚀
    cv::dilate(combinedImage, combinedImage, cv::Mat(), cv::Point(-1, -1), 2);
    cv::erode(combinedImage, combinedImage, cv::Mat(), cv::Point(-1, -1), 2);

    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(combinedImage.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 过滤出符合条件的轮廓（示例条件：轮廓面积大于阈值）
    std::vector<Armor> detectedArmors;
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area > 100) {
            // 创建装甲板对象并存储其矩形边界框
            Armor armor;
            armor.boundingRect = cv::boundingRect(contour);
            detectedArmors.push_back(armor);
        }
    }

    return detectedArmors;
}

// 光条检测函数
void detectLightBars(const cv::Mat& src, std::vector<cv::Rect>& lightBars) {
    // 转换为灰度图像
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    // 高斯滤波
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);

    // 边缘检测
    cv::Canny(gray, gray, 50, 150);

    // 膨胀和腐蚀
    cv::dilate(gray, gray, cv::Mat(), cv::Point(-1, -1), 2);
    cv::erode(gray, gray, cv::Mat(), cv::Point(-1, -1), 2);

    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(gray.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 过滤出符合条件的轮廓（示例条件：轮廓面积大于阈值）
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area > 100) {
            cv::Rect boundingRect = cv::boundingRect(contour);
            lightBars.push_back(boundingRect);
        }
    }
}

int main() {
    cv::VideoCapture cap("/home/lawrence/test/red_light_tracking/ood_red.mp4");

    if (!cap.isOpened()) {
        std::cerr << "Error opening video file." << std::endl;
        return -1;
    }

    cv::Mat frame;
    double fps;
    std::chrono::steady_clock::time_point start, end;
    int frameCount = 0;

    // 设置二值化阈值
    const int binaryThreshold = 128;

    while (cap.read(frame)) {
        // 计算FPS
        if (frameCount == 0) {
            start = std::chrono::steady_clock::now();
        }

        // 二值化图像
        cv::Mat binaryImage = binarizeImage(frame, binaryThreshold);

        // 在左下角显示二值化图像
        cv::Rect roi(0, frame.rows - binaryImage.rows, binaryImage.cols, binaryImage.rows);
        binaryImage.copyTo(frame(roi));

        // 装甲板检测
        std::vector<Armor> detectedArmors = detectArmors(binaryImage, frame);

        // 在左上角显示FPS
        if (frameCount > 0) {
            end = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsedSeconds = end - start;
            fps = static_cast<double>(frameCount) / elapsedSeconds.count();
        }

        cv::putText(frame, "FPS: " + std::to_string(static_cast<int>(fps)), cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);

        // 在右上角显示检测到的装甲板
        for (const auto& armor : detectedArmors) {
            cv::rectangle(frame, armor.boundingRect, cv::Scalar(0, 0, 255), 2);
        }

        // 显示结果
        cv::imshow("Armor Detection", frame);

        // 按ESC键退出
        if (cv::waitKey(30) == 27) {
            break;
        }

        frameCount++;
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
