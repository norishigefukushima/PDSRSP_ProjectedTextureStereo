#include "projection.hpp"
#include <inlineSIMDFunctions.hpp>

using namespace cv;

void addRandmizedSatellitePoints(const Mat& mask, Mat& dest, RNG& rng)
{
	Mat dst;
	if (mask.data == dest.data) dst.create(mask.size(), CV_8U);
	else dst = dest;

	mask.copyTo(dst);

	for (int j = 1; j < mask.rows - 1; j++)
	{
		const uchar* ms = mask.ptr<uchar>(j);
		uchar* m = dst.ptr<uchar>(j);
		const int u = -mask.cols;
		const int d = mask.cols;
		for (int i = 1; i < mask.cols - 1; i++)
		{
			if (ms[i] != 0)
			{
				const int idx = rng.uniform(0, 26);
				switch (idx)
				{
				case 1:m[i + 1] = 255; break;
				case 2:m[i + d - 1] = 255; break;
				case 3:m[i + d + 0] = 255; break;
				case 4:m[i + d + 1] = 255; break;
				case 5:m[i + d] = 255; m[i + u] = 255; break;
				case 6:m[i + 1] = 255; m[i - 1] = 255; break;

				case 7:m[i + d] = 255; m[i - 1] = 255; break;
				case 8:m[i + d] = 255; m[i + 1] = 255; break;
				case 9:m[i + u] = 255; m[i - 1] = 255; break;
				case 10:m[i + u] = 255; m[i + 1] = 255; break;
				case 11:m[i + u - 1] = 255; m[i + d + 1] = 255; break;
				case 12:m[i + u + 1] = 255; m[i + d - 1] = 255; break;
				case 13:m[i + u - 1] = 255; m[i + u + 1] = 255; break;

				case 14:m[i + u + 1] = 255; m[i + d + 1] = 255; break;
				case 15:m[i + d - 1] = 255; m[i + d + 1] = 255; break;
				case 16:m[i + u - 1] = 255; m[i + d - 1] = 255; break;
				case 17:m[i + u - 1] = 255; m[i + d] = 255; break;
				case 18:m[i + u + 1] = 255; m[i + d] = 255; break;
				case 19:m[i - 1] = 255; m[i + u + 1] = 255; break;
				case 20:m[i - 1] = 255; m[i + d + 1] = 255; break;

				case 21:m[i + u] = 255; m[i + d - 1] = 255; break;
				case 22:m[i + u] = 255; m[i + d + 1] = 255; break;
				case 23:m[i + u - 1] = 255; m[i + 1] = 255; break;
				case 24:m[i + d - 1] = 255; m[i + 1] = 255; break;
				}
			}
		}
	}
	if (mask.data == dest.data)dst.copyTo(mask);
}

void applyProjectionWeight_I8UC3_W32FC1(const Mat& image, const Mat& weight, const float dark_amp, Mat& dest)
{
	CV_Assert(image.type() == CV_8UC3);
	CV_Assert(weight.type() == CV_32FC1);
	dest.create(image.size(), image.type());

	const uchar* im = image.ptr<uchar>();
	const float* wm = weight.ptr<float>();
	uchar* d = dest.ptr<uchar>();

	const int size = image.size().area();
	const int SIZE = get_simd_floor(size, 8);
	const float ida = 1.f - dark_amp;
	const __m256 mida = _mm256_set1_ps(ida);
	const __m256 mone = _mm256_set1_ps(1.f);
	
	__m256 b, g, r;
	for (int i = 0; i < SIZE; i += 8)
	{
		const __m256 mw = _mm256_fnmadd_ps(mida, _mm256_min_ps(_mm256_loadu_ps(wm + i), mone),mone);
		_mm256_load_cvtepu8bgr2planar_ps(im + 3 * i, b, g, r);
		b = _mm256_mul_ps(mw, b);
		g = _mm256_mul_ps(mw, g);
		r = _mm256_mul_ps(mw, r);
		_mm256_store_ps2epu8_color(d + 3 * i, b, g, r);
	}
	for (int i = SIZE; i < size; i++)
	{
		const float w = 1.f - ida * min(wm[i], 1.f);
		d[3 * i + 0] = saturate_cast<uchar>(im[3 * i + 0] * w);
		d[3 * i + 1] = saturate_cast<uchar>(im[3 * i + 1] * w);
		d[3 * i + 2] = saturate_cast<uchar>(im[3 * i + 2] * w);
	}
}

void projectPattern(const Mat& leftImage, Mat& destLeftImage, const Mat& rightImage, Mat& destRightImage, const Mat& centerDisparityMap32F, const Mat& pattern, const float mask_size, const float dark_amp)
{
	CV_Assert(centerDisparityMap32F.depth() == CV_32F);

	AutoBuffer<float> dbuffL(leftImage.cols);
	AutoBuffer<float> dbuffR(leftImage.cols);
	Mat darkL = Mat::zeros(leftImage.size(), CV_32F);
	Mat darkR = Mat::zeros(leftImage.size(), CV_32F);
	
	const bool fixCenterDisp = true;//for projection with constant disparity map

	for (int j = 1; j < leftImage.rows - 1; j++)
	{
		float* dl = darkL.ptr<float>(j);
		float* dr = darkR.ptr<float>(j);
		const float* disp = centerDisparityMap32F.ptr<float>(j);

		memset(dbuffL, 0, sizeof(float) * leftImage.cols);
		memset(dbuffR, 0, sizeof(float) * leftImage.cols);
		for (int i = 2; i < leftImage.cols - 2; i++)
		{
			const float d = (disp[i]);
			const int dh = int(d * 0.5f);
			if ((i + dh) < leftImage.cols)if (dbuffL[i + dh] < d) dbuffL[i + dh] = d;
			if ((i - dh) >= 0)if (dbuffR[i - dh] < d) dbuffR[i - dh] = d;
		}

		if (fixCenterDisp)
		{
			for (int i = 2; i < leftImage.cols - 2; i++)
			{
				const float d = disp[i];
				const float df = d * 0.5f;
				const int dh = int(d * 0.5f);
				if (d != 0 && pattern.at<uchar>(j, i) != 0 && (i - dh - 2) >= 0 && (i + dh + 2) < leftImage.cols)
				{
					const float _a = df - dh;
					const float ia = 1.f - _a;
					if (dbuffL[i + dh] == d)
					{
						for (int l = -1; l <= 1; l++)
						{
							const float wv = (l == 0) ? 1.f : mask_size - 1.f;
							for (int k = -1; k <= 1; k++)
							{
								const float wh = (k == 0) ? 1.f : mask_size - 1.f;
								darkL.at<float>(j + l, i + k + dh + 0) += ia * wh * wv;
								darkL.at<float>(j + l, i + k + dh + 1) += _a * wh * wv;
							}
						}
					}
					if (dbuffR[i - dh] == d)
					{
						for (int l = -1; l <= 1; l++)
						{
							const float wv = (l == 0) ? 1.f : mask_size - 1.f;
							for (int k = -1; k <= 1; k++)
							{
								const float wh = (k == 0) ? 1.f : mask_size - 1.f;
								darkR.at<float>(j + l, i + k - dh - 0) += ia * wh * wv;
								darkR.at<float>(j + l, i + k - dh - 1) += _a * wh * wv;
							}
						}
					}
				}
			}
		}
		else
		{
			for (int i = 2; i < leftImage.cols - 2; i++)
			{
				const float d = disp[i];
				const float df = d * 0.5f;
				const int dh = int(d * 0.5f);
				if (d != 0 && pattern.at<uchar>(j, i) != 0 && (i - dh - 2) >= 0 && (i + dh + 2) < leftImage.cols)
				{
					if (dbuffL[i + dh] == d)
					{
						for (int l = -1; l <= 1; l++)
						{
							const float wv = (l == 0) ? 1.f : mask_size - 1.f;
							for (int k = -1; k <= 1; k++)
							{
								const float d = centerDisparityMap32F.at<float>(j + l, i + k);
								const float df = d * 0.5f;
								const int dh = int(d * 0.5f);
								const float _a = df - dh;
								const float ia = 1.f - _a;
								const float wh = (k == 0) ? 1.f : mask_size - 1.f;
								darkL.at<float>(j + l, i + k + dh + 0) += ia * wh * wv;
								darkL.at<float>(j + l, i + k + dh + 1) += _a * wh * wv;
							}
						}
					}
					if (dbuffR[i - dh] == d)
					{
						for (int l = -1; l <= 1; l++)
						{
							const float wv = (l == 0) ? 1.f : mask_size - 1.f;
							for (int k = -1; k <= 1; k++)
							{
								const float d = centerDisparityMap32F.at<float>(j + l, i + k);
								const float df = d * 0.5f;
								const int dh = int(d * 0.5f);
								const float _a = df - dh;
								const float ia = 1.f - _a;
								const float wh = (k == 0) ? 1.f : mask_size - 1.f;
								darkR.at<float>(j + l, i + k - dh - 0) += ia * wh * wv;
								darkR.at<float>(j + l, i + k - dh - 1) += _a * wh * wv;
							}
						}
					}
				}
			}
		}
	}

	applyProjectionWeight_I8UC3_W32FC1(leftImage, darkL, dark_amp, destLeftImage);
	applyProjectionWeight_I8UC3_W32FC1(rightImage, darkR, dark_amp, destRightImage);
}