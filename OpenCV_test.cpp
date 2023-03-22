#include <opencv2/opencv.hpp>
#include<opencv2/core/mat.hpp>
#include<opencv2/core/mat.inl.hpp>
#include<iostream>
#include<math.h>

using namespace std;
using namespace cv;

#define COLOR_MIN 0
#define COLOR_MAX 255
#define pi 3.1415926

double** Gussion_template(int ksize, double sigma, double** gen_template)
{
	int center = ksize / 2;
	double ratio1 = 1 / (2 * pi * sigma * sigma);
	double ratio2 = 1 / (2 * sigma * sigma);//提取常量，加速计算

	double x, y;
	for (int i = 0; i < ksize; i++)
	{
		x = pow(i - center, 2);
		for (int j = 0; j < ksize; j++)
		{
			y = pow(j - center, 2);
			gen_template[i][j] = ratio1 * exp(-(x + y) * ratio2);
		}
	}

	double k = 1 / gen_template[0][0];
	for (int i = 0; i < ksize; i++)
	{
		for (int j = 0; j < ksize; j++)
		{
			gen_template[i][j] *= k;
		}
	}
	return gen_template;
}

void Gaussion_filter(Mat& src_img, Mat& dst_img, int ksize, double sigma)
{

	CV_Assert(src_img.channels() == 1 || src_img.channels() == 3);

	double** gen_template = new double* [ksize];//把数组定义在这里是为了能析构，防止内存泄露
	for (int i = 0; i < ksize; i++)
	{
		gen_template[i] = new double[ksize];
	}
	gen_template = Gussion_template(ksize, sigma, gen_template);

	cout << "————————————————————————————————————————My Gaussion filter";
	cout << "————————————————————————————————————————" << endl;
	double total = 0;
	for (int i = 0; i < ksize; i++)
	{
		for (int j = 0; j < ksize; j++)
		{
			total += gen_template[i][j];
			cout << setiosflags(ios::fixed) << setprecision(1) << gen_template[i][j] << " ";
		}
		cout << endl;
	}

	cout << setiosflags(ios::left) << setw(30) << "size before make border" << src_img.cols << " " << src_img.rows << endl;
	int border = ksize / 2;
	copyMakeBorder(src_img, src_img, border, border, border, border, BorderTypes::BORDER_CONSTANT);
	cout << setiosflags(ios::left) << setw(30) << "size after make border" << src_img.cols << " " << src_img.rows << endl;

	for (int i = 0; i <= src_img.rows - ksize; i++)//等价于i <= src_img.rows - ksize + 1
	{
		for (int j = 0; j <= src_img.cols - ksize; j++)
		{
			if (src_img.channels() == 1)//实现了灰度图像处理
			{
				dst_img.at<double>(i, j) = 0;
				for (int k = 0; k < ksize; k++)
				{
					for (int l = 0; l < ksize; l++)
					{
						dst_img.at<double>(i, j) += src_img.at<double>(i + k, j + l) * gen_template[k][l];
					}
				}
				dst_img.at<double>(i, j) /= total;
				if (dst_img.at<double >(i, j) / total > 255)
				{
					dst_img.at<double>(i, j) = 255;
				}
			}
			else if (src_img.channels() == 3)
			{
				double rgb_temp[3] = { 0 };
				for (int k = 0; k < ksize; k++)
				{
					for (int l = 0; l < ksize; l++)
					{
						Vec3b rgb = src_img.at<Vec3b>(i + k, j + l);
						for (int m = 0; m < 3; m++)
						{
							rgb_temp[m] += rgb[m] * gen_template[k][l];
						}
					}
				}
				for (int m = 0; m < 3; m++)
				{
					rgb_temp[m] /= total;
					if (rgb_temp[m] > 255) rgb_temp[m] = 255;
					dst_img.at<Vec3b>(i, j)[m] = static_cast<uchar>(rgb_temp[m]);
				}
			}
		}
	}
	cout << setiosflags(ios::left) << setw(30) << "size of dst image" << dst_img.cols << " " << dst_img.rows << endl;
	for (int i = 0; i < ksize; i++)
	{
		delete gen_template[i];
	}
	delete gen_template;
}

void DeleteOneColOfMat(Mat& object, int num)
{
	if (num < 0 || num >= object.cols)
	{
		cout << " Col overflow " << endl;
	}
	else
	{
		if (num == object.cols - 1)
		{
			object = object.t();
			object.pop_back();
			object = object.t();
		}
		else
		{
			for (int i = num + 1; i < object.cols; i++)
			{
				object.col(i - 1) = object.col(i) + Scalar(0, 0, 0, 0);
			}
			object = object.t();
			object.pop_back();
			object = object.t();
		}
	}
}

void DeleteOneRowOfMat(Mat& object, int num)
{
	if (num < 0 || num >= object.rows)
	{
		cout << " Row overflow " << endl;
	}
	else
	{
		if (num == object.rows - 1)
		{
			object.pop_back();
		}
		else
		{
			for (int i = num + 1; i < object.rows; i++)
			{
				object.row(i - 1) = object.row(i) + Scalar(0, 0, 0, 0);
			}
			object.pop_back();
		}
	}
}

void remove_border(Mat& image, int ksize)
{
	cout << setiosflags(ios::left) << setw(30) << "size before remove border" << image.cols << " " << image.rows << endl;
	for (int i = 0; i < ksize / 2; i++)
	{

		DeleteOneColOfMat(image, image.cols - 1);
		DeleteOneColOfMat(image, 0);
		DeleteOneRowOfMat(image, image.rows - 1);
		DeleteOneRowOfMat(image, 0);
	}
	cout << setiosflags(ios::left) << setw(30) << "size after remove border"  << image.cols << " " << image.rows << endl;
}

int main()
{
	Mat image_low_src = imread("1.png");
	Mat image_low = imread("1.png");
	Mat image_high_src = imread("2.png");
	Mat image_high = imread("2.png");

	resize(image_high_src, image_high_src, Size(image_low_src.cols, image_low_src.rows));
	resize(image_high, image_high, Size(image_low.cols, image_low.rows));

	int ksize = 25;
	double sigma = 8;

	/*GaussianBlur(image_high_src, image_high, Size(ksize, ksize), sigma, sigma);
	image_high = image_high_src - image_high;
	namedWindow("1", WINDOW_NORMAL);
	imshow("1", image_high);
	GaussianBlur(image_high_src, image_high, Size(ksize * 2 + 1, ksize * 2 + 1), sigma, sigma);
	image_high = image_high_src - image_high;
	namedWindow("2", WINDOW_NORMAL);
	imshow("2", image_high);
	GaussianBlur(image_high_src, image_high, Size(ksize / 2 - 1, ksize / 2 - 1), sigma, sigma);
	image_high = image_high_src - image_high;
	namedWindow("3", WINDOW_NORMAL);
	imshow("3", image_high);
	waitKey();*/

	Gaussion_filter(image_low_src, image_low, ksize, sigma);
	namedWindow("Low frequency", WINDOW_NORMAL);
	imshow("Low frequency", image_low);
	waitKey();

	Gaussion_filter(image_high_src, image_high, ksize, sigma);
	remove_border(image_high_src, ksize);

	int color_change = 12;
	for (int i = 0; i < image_high_src.rows; i++)
	{
		for (int j = 0; j < image_high_src.cols; j++)
		{
			int rgb_temp[3] = { 0 };
			Vec3b rgb_src = image_high_src.at<Vec3b>(i, j);
			Vec3b rgb_dst = image_high.at<Vec3b>(i, j);

			for (int k = 0; k < 3; k++)
			{
				rgb_temp[k] += rgb_src[k] + color_change - rgb_dst[k];
				rgb_temp[k] = min(rgb_temp[k], COLOR_MAX);
				rgb_temp[k] = max(rgb_temp[k], COLOR_MIN);
				image_high.at<Vec3b>(i, j)[k] = static_cast<uchar>(rgb_temp[k]);
			}
		}
	}
	//image_high = image_high_src - image_high;

	namedWindow("High frequency", WINDOW_NORMAL);
	imshow("High frequency", image_high);
	waitKey();

	Mat result;
	result = image_low + image_high;
	namedWindow("Hybrid image", WINDOW_NORMAL);
	imshow("Hybrid image", result);
	waitKey();
	system("pause");
}
