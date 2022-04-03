#include <opencp.hpp>
using namespace std;
using namespace cv;
using namespace cp;

void PointCloudRenderingProp(const float a, const float b, const int color)
{
	CSV csv("pl/test3DStereo.csv", false, false);
	csv.readData();

	Mat prop_(Size((int)csv.data[0].size(), (int)csv.data.size()), CV_32F);

	for (int j = 0; j < prop_.rows; j++)
	{
		for (int i = 0; i < prop_.cols; i++)
		{
			const float v = float(csv.data[j][i]);
			prop_.at<float>(j, i) = v > 1200.f ? 10000.f : v;
		}
	}
	Mat prop = prop_(Rect(14, 10, 1034, 980 - 80)).clone();
	cp::binaryWeightedRangeFilter(prop, prop, Size(13, 13), 2, 2);
	Mat propdisp;

	prop.convertTo(propdisp, CV_8U, a, -b * a);

	Mat propim;
	applyColorMap(propdisp, propim, color);
	//guiAlphaBlend(propdisp, propim);
	{
		cp::PointCloudShow pcs;
		pcs.loopDepth(propim, prop, 5544.3f, 0);
	}
}

void PointCloudRenderingRealSense(const float a, const float b, const int color)
{
	FILE* fp = fopen("pl/3a_Depth.raw", "rb");
	AutoBuffer<short> data(320 * 240);
	fread(data, sizeof(short), 320 * 240, fp);
	fclose(fp);

	Mat depthRealSense(Size(320, 240), CV_16S, data);

	for (int i = 0; i < depthRealSense.size().area(); i++)
	{
		depthRealSense.at<short>(i) = (depthRealSense.at<short>(i) == 0) ? 10000 : depthRealSense.at<short>(i);
	}

	resize(depthRealSense, depthRealSense, Size(), 4, 4, INTER_NEAREST);
	Mat imRealSense = convert(depthRealSense, CV_8U, a, -b * a);

	Mat im; applyColorMap(imRealSense, im, color);
	//guiAlphaBlend(imRealSense, im);

	cp::PointCloudShow pcs;
	pcs.loopDepth(im, depthRealSense, 640, 0);
}