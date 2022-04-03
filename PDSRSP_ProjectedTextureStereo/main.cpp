#include <opencp.hpp>
#pragma comment(lib, "opencp.lib")
#include "stereo.hpp"
#include "projection.hpp"

using namespace std;
using namespace cv;
using namespace cp;

void PointCloudRenderingProp(const float a, const float b, const int color);
void PointCloudRenderingRealSense(const float a, const float b, const int color);

void generatePDSMask(const Size size, const float d, const int numMasks)
{
#pragma omp parallel for
	for (int n = 0; n < numMasks; n++)
	{
		RNG rng(getTickCount());
		Timer t;

		Mat mask(size, CV_8U);
		PoissonDiskSampling pds(d, mask.size());
		pds.generate(mask, rng);
		cout << float(countNonZero(mask)) / mask.size().area() * 100.0 << endl;
		imwrite(format("PDSmask_d%02d_%02d.png", int(d * 10), n), mask);
	}
}

void pointcloudTest()
{
	PointCloudRenderingProp(255.f / 70.f, 760.f, COLORMAP_AUTUMN);
	PointCloudRenderingRealSense(255.f / 70.f, 285.f, COLORMAP_AUTUMN);
}

void stereoTest()
{
	const bool isAddNoise = false;
	const float sigma = 5.f;
	const float gamma = 1.2f;

	Mat disp;
	vector<string> fname = {
	"Aloe",//0
	"Art",//1
	"Baby1",//2
	"Baby2",//3
	"Baby3",//4
	"Books",//5
	"Bowling1",//6
	"Bowling2",//7
	"Cloth1",//8
	"Cloth2",//9
	"Cloth3",//10
	"Cloth4",//11
	"Dolls",//12
	"Flowerpots",//13
	"Lampshade1",//14
	"Lampshade2",//15
	"Laundry",//16
	"Midd1",//17
	"Midd2",//18
	"Moebius",//19
	"Monopoly",//20
	"Plastic",//21
	"Reindeer",//22
	"Rocks1",//23
	"Rocks2",//24
	"Wood1",//25
	"Wood2"//26
	};

	vector<int>doffset{
		270,200,300,300,250,
		200,290,240,290,260,
		290,260,200,251,260,
		260,230,196,214,200,
		237,280,230,274,274,
		210,254
	};

	csv.write(" ");
	csv.write("CENSUS13x3 NBP");
	csv.write("CENSUS13x3 MSE");
	csv.write("CENSUS7x3 NBP");
	csv.write("CENSUS7x3 MSE");
	csv.write("AD NBP");
	csv.write("AD MSE");
	csv.write("SD NBP");
	csv.write("SD MSE");
	csv.write("ADEdge NBP");
	csv.write("ADEdge MSE");
	csv.write("SDEdge NBP");
	csv.write("SDEdge MSE");
	csv.end();

	//i=21
	for (int i = 0; i < 27; i++)
	{
		const int amp = 1;
		string name = fname[i];
		cout << name << endl;
		Mat left_ = imread("img/" + name + "/view1.png");
		Mat right_ = imread("img/" + name + "/view5.png");
		Mat dmapL_ = imread("img/" + name + "/disp1.png", 0);
		Mat dmapR_ = imread("img/" + name + "/disp5.png", 0);

		Mat left = left_;
		Mat right = right_;
		Mat dmapL = dmapL_;
		Mat dmapR = dmapR_;
		const bool isPad = false;
		if (isPad)
		{
			int pad = get_simd_ceil(left_.cols, 16) - left_.cols;
			copyMakeBorder(left_, left, 0, 0, 0, pad, cv::BORDER_DEFAULT);
			copyMakeBorder(right_, right, 0, 0, 0, pad, cv::BORDER_DEFAULT);
			copyMakeBorder(dmapL_, dmapL, 0, 0, 0, pad, cv::BORDER_DEFAULT);
			copyMakeBorder(dmapR_, dmapR, 0, 0, 0, pad, cv::BORDER_DEFAULT);
		}

		int disp_min, disp_max;
		cp::computeDisparityMinMax(dmapL, amp, disp_min, disp_max);
		disp_min = get_simd_floor(disp_min, 16);
		disp_max = get_simd_ceil(disp_max, 16);
		cp::StereoEval eval(dmapL, amp, disp_max, 100);

		StereoMatch sm(5, disp_min, disp_max);
		sm.setRefinementMethod(StereoMatch::REFINEMENT::NONE);

		Mat dmapC;
		cp::generateCenterDisparity(dmapL, dmapR, dmapC, amp, CV_32F);

		if (isAddNoise)
		{
			addNoise(left, left, sigma);
			addNoise(right, right, sigma);
			cp::contrastGamma(right, right, gamma);
			//cp::contrastSToneExp(right, right, 30, 0.5);
		}

		csv.write(fname[i]);
		sm.gui(left, right, disp, eval, dmapC, doffset[i]);
		csv.end();
	}
}

int main()
{
	stereoTest();
	//generatePDS(Size(1400, 1120), 2.f, 10);
	pointcloudTest();
	
	return 0;
}