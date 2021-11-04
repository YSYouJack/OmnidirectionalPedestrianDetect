#include <filesystem>
#include <iostream>
#include <string>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

namespace fs = std::filesystem;

static bool isSupportImg(const fs::path& filePath)
{
	static const std::string VALID_IMAGE_EXT[5] = {
		".bmp", ".jpg", ".jpeg", ".png", ".tif"
	};

#if _MSC_VER >= 1910
	std::string ext = filePath.extension().string();
#else
	std::string ext = filePath.extension();
#endif
	for (auto& c : ext) {
		c = static_cast<char>(std::tolower(c));
	}

	for (size_t i = 0; i < 5; ++i) {
		if (ext == VALID_IMAGE_EXT[i]) {
			return true;
		}
	}

	return false;
}

int main(int argc, char** argv)
{
	if (2 != argc) {
		std::cout << "Usage: ImagesToVideo image_folder" << std::endl;
		return 1;
	}

	fs::path inDir = argv[1];
	if (!fs::is_directory(inDir)) {
		std::cerr << "Error: The " << inDir << " is not a directory!" << std::endl;
		return 1;
	}

	// Scan for input size.	
	std::vector<fs::path> images;
	fs::directory_iterator end;
	fs::directory_iterator it(inDir);
	for (; it != end; ++it) {
		if (fs::is_directory(it->status())) {
			continue;
		} else if (isSupportImg(it->path())) {
			images.emplace_back(it->path());
		}
	}

	if (images.empty()) {
		std::cerr << "Error: No image found!" << std::endl;
		return 1;
	}

	cv::Mat frame = cv::imread(images[0].string(), cv::IMREAD_UNCHANGED);

	// Open video writer.
	int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
	fs::path outFileName = inDir / fs::path("video.avi");

	//int fourcc = cv::VideoWriter::fourcc('X', '2', '6', '4');
	//fs::path outFileName = inDir / fs::path("video.mp4");

	double fps = 10.0;

	cv::VideoWriter vw;
	if (!vw.open(outFileName.string(), fourcc, fps, frame.size(), CV_8UC3 == frame.type())) {
		std::cerr << "Error: Unsupported video format" << std::endl;
		return 1;
	}
	vw.write(frame);
	
	// Write frame.
	for (size_t i = 0; i < images.size(); ++i) {
		if (0 != i) {
			frame = cv::imread(images[i].string(), cv::IMREAD_UNCHANGED);
		}
		std::cout << "Writing frame " << i + 1 << "/" << images.size() << "." << std::endl;
		vw.write(frame);
	}

    return 0;
}