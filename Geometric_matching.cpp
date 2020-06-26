#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/cudaimgproc.hpp"

using namespace std;
using namespace cv;

const bool use_gpu = true;

int main()
{
	Mat img_template = imread("template_small_2.png", IMREAD_GRAYSCALE);
	Mat img = imread("image.png", IMREAD_GRAYSCALE);

	Ptr<GeneralizedHoughGuil> guil = use_gpu ? cuda::createGeneralizedHoughGuil() : createGeneralizedHoughGuil();
	
	guil->setMinDist(100);
	guil->setLevels(360);
	guil->setDp(4);
	guil->setMaxBufferSize(4000);

	guil->setMinAngle(0);
	guil->setMaxAngle(360);
	guil->setAngleStep(1);
	guil->setAngleThresh(15000);

	guil->setMinScale(0.9);
	guil->setMaxScale(1.1);
	guil->setScaleStep(0.05);
	guil->setScaleThresh(1000);

	guil->setPosThresh(400);
	//guil->setCannyHighThresh(230);
	//guil->setCannyLowThresh(150);

	double sobelScale = 0.05;
	int sobelKernelSize = 5;
	int cannyHigh = 230;
	int cannyLow = 150;

	Mat canny;
	Canny(img, canny, cannyHigh, cannyLow);
	namedWindow("canny", WINDOW_NORMAL);
	imshow("canny", canny);

	Mat dx;
	Sobel(img, dx, CV_32F, 1, 0, sobelKernelSize, sobelScale);
	namedWindow("dx", WINDOW_NORMAL);
	imshow("dx", dx);

	Mat dy;
	Sobel(img, dy, CV_32F, 0, 1, sobelKernelSize, sobelScale);
	namedWindow("dy", WINDOW_NORMAL);
	imshow("dy", dy);

	Mat canny_template;
	Canny(img_template, canny_template, cannyHigh, cannyLow);
	namedWindow("canny_template", WINDOW_NORMAL);
	imshow("canny_template", canny_template);

	Mat dx_template;
	Sobel(img_template, dx_template, CV_32F, 1, 0, sobelKernelSize, sobelScale);
	namedWindow("dx_template", WINDOW_NORMAL);
	imshow("dx_template", dx_template);

	Mat dy_template;
	Sobel(img_template, dy_template, CV_32F, 0, 1, sobelKernelSize, sobelScale);
	namedWindow("dy_template", WINDOW_NORMAL);
	imshow("dy_template", dy_template);

	vector<Vec4f> position;
	TickMeter tm;
	Mat votes;

	if (use_gpu) {
		cuda::GpuMat d_template(img_template);
		cuda::GpuMat d_edges_template(canny_template);
		cuda::GpuMat d_image(img);
		cuda::GpuMat d_x(dx);
		cuda::GpuMat d_dx_template(dx_template);
		cuda::GpuMat d_y(dy);
		cuda::GpuMat d_dy_template(dy_template);
		cuda::GpuMat d_canny(canny);
		cuda::GpuMat d_position;
		cuda::GpuMat d_votes;

		guil->setTemplate(d_edges_template, d_dx_template, d_dy_template);

		tm.start();

		guil->detect(d_canny, d_x, d_y, d_position, d_votes);
		
		if (d_position.size().height != 0) {
			d_position.download(position);
			d_votes.download(votes);
		}
		
		tm.stop();
	}
	else {
		guil->setTemplate(canny_template, dx_template, dy_template);

		tm.start();

		guil->detect(canny, dx, dy, position, votes);

		tm.stop();
	}

	cout << "Found : " << position.size() << " objects" << endl;
	cout << "Detection time : " << tm.getTimeMilli() << " ms" << endl;

	Mat out;
	cvtColor(img, out, COLOR_GRAY2BGR);
	int index = 0;
	Mat votes_int = Mat(votes.rows, votes.cols, CV_32SC3, votes.data, votes.step);
	for (auto& i : position) {
		Point2f pos(i[0], i[1]);
		float scale = i[2];
		float angle = i[3];
		
		RotatedRect rect;
		rect.center = pos;
		rect.size = Size2f(img_template.cols * scale, img_template.rows * scale);
		rect.angle = angle;

		Point2f pts[4];
		rect.points(pts);

		line(out, pts[0], pts[1], Scalar(0, 255, 0), 3);
		line(out, pts[1], pts[2], Scalar(0, 255, 0), 3);
		line(out, pts[2], pts[3], Scalar(0, 255, 0), 3);
		line(out, pts[3], pts[0], Scalar(0, 255, 0), 3);

		int score = votes_int.at<Vec3i>(index).val[0];
		index++;
		string score_string = to_string(score);
		putText(out, score_string, rect.center, FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2);
	}

	namedWindow("out", WINDOW_NORMAL);
	imshow("out", out);
	waitKey();
}