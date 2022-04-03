#include "stereo.hpp"
#include "projection.hpp"
#define GUI_LOOP
//#define TIMER_STEREO_MATCH
using namespace std;
using namespace cv;
using namespace cp;

float loadPrecomputedPDS(Mat& mask, int distanceIndex, int index)
{
	int n = 0;
	switch (distanceIndex)
	{
	case 0:n = 25; break;
	case 1:n = 30; break;
	case 2:n = 35; break;
	case 3:n = 40; break;
	case 4:n = 45; break;
	case 5:n = 50; break;
	case 6:n = 55; break;
	case 7:n = 60; break;
	case 8:n = 65; break;
	case 9:n = 70; break;
	default:
		break;
	}
	const int idx = index;
	Mat temp = imread(format("pds/PDSmask_d%02d_%02d.png", n, idx), 0);
	int x = 2;
	int y = 2;
	temp(Rect(x, y, mask.cols, mask.rows)).copyTo(mask);
	return float(countNonZero(mask)) / mask.size().area();
}

#pragma region public
StereoMatch::StereoMatch(int blockSize, int minDisp, int disparityRange) :thread_max(omp_get_max_threads())
{
	border = 0;
	speckleWindowSize = 20;
	speckleRange = 16;
	uniquenessRatio = 0;

	subpixelInterpolationMethod = (int)SUBPIXEL::QUAD;

	subpixelRangeFilterWindow = 2;
	subpixelRangeFilterCap = 16;

	pixelMatchErrorCap = 31;
	aggregationGuidedfilterEps = 0.1;
	aggregationSigmaSpace = 255.0;
	aggregationRadiusH = blockSize;
	aggregationRadiusV = blockSize;
	minDisparity = minDisp;
	numberOfDisparities = disparityRange;

	preFilterCap = 31;
	costAlphaImageSobel = 10;
	sobelBlendMapParam_Size = 0;
	sobelBlendMapParam1 = 50;
	sobelBlendMapParam2 = 20;

	P1 = 0;
	P2 = 0;
	gif = new cp::GuidedImageFilter[thread_max];
}

StereoMatch::~StereoMatch()
{
	delete[] gif;
}

void StereoMatch::imshowDisparity(string wname, Mat& disp, int option, OutputArray output)
{
	output.create(disp.size(), CV_8UC3);
	Mat dst = output.getMat();
	cvtDisparityColor(disp, dst, 0, 64, (DISPARITY_COLOR)option, 16);
	imshow(wname, output);
}

void StereoMatch::imshowDisparity(string wname, Mat& disp, int option, Mat& output, int mindis, int range)
{
	cvtDisparityColor(disp, output, mindis, range, (DISPARITY_COLOR)option, 16);
	imshow(wname, output);
}

//main function
void StereoMatch::matching(Mat& leftim, Mat& rightim, Mat& destDisparityMap, const bool isFeedback)
{
#ifdef TIMER_STEREO_MATCH
	Timer t("total");
#endif
	if (destDisparityMap.empty() || leftim.size() != destDisparityMap.size()) destDisparityMap.create(leftim.size(), CV_16S);
	minCostMap.create(leftim.size(), CV_8U);
	minCostMap.setTo(255);
	if ((int)DSI.size() < numberOfDisparities)DSI.resize(numberOfDisparities);

#pragma region matching:prefilter
	computeGuideImageForAggregation(leftim);

	{
#ifdef TIMER_STEREO_MATCH
		Timer t("pre filter");
#endif
		computePrefilter(leftim, rightim);
	}
#pragma endregion

#pragma region matching: pixelmatch and aggregation
	{
#ifdef TIMER_STEREO_MATCH
		Timer t("Cost computation & aggregation");
#endif
		if (aggregationMethod == CrossBasedBox) clf.makeKernel(guideImage, aggregationRadiusH, (int)aggregationGuidedfilterEps, 0);

#pragma omp parallel for schedule (dynamic)
		for (int i = 0; i < numberOfDisparities; i++)
		{
			const int d = minDisparity + i;
			computePixelMatchingCost(d, DSI[i]);

			if (isFeedback)addCostIterativeFeedback(DSI[i], d, destDisparityMap, feedbackFunction, feedbackClip, feedbackAmp);
			computeCostAggregation(DSI[i], DSI[i], guideImage);
		}
	}
#pragma endregion

#pragma region matching: optimization and WTA
	{
#ifdef TIMER_STEREO_MATCH
		Timer t("Cost Optimization");
#endif
		if (P1 != 0 && P2 != 0)
			computeOptimizeScanline();
	}

	{
#ifdef TIMER_STEREO_MATCH
		Timer t("DisparityComputation");
#endif
		computeWTA(DSI, destDisparityMap, minCostMap);
	}
#pragma endregion

#pragma region matching:postfiltering
	{
#ifdef TIMER_STEREO_MATCH
		Timer t("===Post Filterings===");
#endif
		{
#ifdef TIMER_STEREO_MATCH
			Timer t("Post: uniqueness");
#endif
			uniquenessFilter(minCostMap, destDisparityMap);
		}
		//subpix;
		{
#ifdef TIMER_STEREO_MATCH
			Timer t("Post: subpix");
#endif
			subpixelInterpolation(destDisparityMap, (SUBPIXEL)subpixelInterpolationMethod);
			if (isRangeFilterSubpix) binaryWeightedRangeFilter(destDisparityMap, destDisparityMap, subpixelRangeFilterWindow, (float)subpixelRangeFilterCap);
		}

		//LR depth map;
		{
#ifdef TIMER_STEREO_MATCH
			Timer t("Post: LR");
#endif
			if (LRCheckMethod == (int)LRCHECK::WITH_MINCOST)
				fastLRCheck(minCostMap, destDisparityMap);
			else if (LRCheckMethod == (int)LRCHECK::WITHOUT_MINCOST)
				fastLRCheck(destDisparityMap);
		}
		{
#ifdef TIMER_STEREO_MATCH
			Timer t("Post: mincost");
#endif
			if (isMinCostFilter)
			{
				minCostThresholdFilter(minCostMap, destDisparityMap, minCostThreshold);
				//minCostSwapFilter(minCostMap, destDisparityMap);
			}
		}
		{
#ifdef TIMER_STEREO_MATCH
			Timer t("Post: filterSpeckles");
#endif
			if (isSpeckleFilter)
				filterSpeckles(destDisparityMap, 0, speckleWindowSize, speckleRange, specklebuffer);
		}

		{
			computeValidRatio(destDisparityMap);

			int occsearch2 = 4;
			int occth = 17;
			int occsearch = 4;
			int occsearchh = 2;
			int occiter = 0;
			if (holeFillingMethod != 0)
			{
#ifdef TIMER_STEREO_MATCH
				Timer t("occ");
#endif
				//fillOcclusion(destDisparity, (minDisparity - 1) * 16);
				fillOcclusion(destDisparityMap);
			}
		}
#pragma region refinement

		{
#ifdef TIMER_STEREO_MATCH
			Timer t("Post: refinement");
#endif
			int rrad = 5;
			int ss = 32;
			int sr1 = 32;
			int sr2 = 32;
			/*static int rrad = 5; createTrackbar("rrad", "", &rrad, 100);
			static  int ss = 32; createTrackbar("ss", "", &ss, 10000);
			static  int sr1 = 32; createTrackbar("sr1", "", &sr1, 250);
			static  int sr2 = 32; createTrackbar("sr2", "", &sr2, 250);*/

			if (refinementMethod == (int)REFINEMENT::GIF_JNF)
			{
				//crossBasedAdaptiveBoxFilter(destDisparity, leftim, destDisparity, Size(2 * gr + 1, 2 * gr + 1), ge);

				Mat temp;
				guidedImageFilter(destDisparityMap, leftim, temp, refinementR, (float)refinementSigmaRange, GUIDED_SEP_VHI);
				jointNearestFilter(temp, destDisparityMap, Size(2 * jointNearestR + 1, 2 * jointNearestR + 1), destDisparityMap);
			}
			else if (refinementMethod == (int)REFINEMENT::WGIF_GAUSS_JNF)
			{
				weightMap.create(leftim.size(), CV_32F);
				const Size ksize = Size(2 * refinementWeightR + 1, 2 * refinementWeightR + 1);

				Mat bim;
				GaussianBlur(destDisparityMap, bim, ksize, refinementWeightR / 3.0);
				short* disp = destDisparityMap.ptr<short>(0);
				short* dispb = bim.ptr<short>();
				float* s = weightMap.ptr<float>();
				for (int i = 0; i < weightMap.size().area(); i++)
				{
					float diff = (disp[i] - dispb[i]) * (disp[i] - dispb[i]) / (-2.f * 16 * 16 * refinementWeightSigma * refinementWeightSigma);
					s[i] = exp(diff);
				}

				Mat a, b;
				multiply(destDisparityMap, weightMap, a, 1, CV_32F);
				guidedImageFilter(a, leftim, a, refinementR, refinementSigmaRange, GUIDED_SEP_VHI);
				guidedImageFilter(weightMap, leftim, b, refinementR, refinementSigmaRange, GUIDED_SEP_VHI);
				Mat temp;
				divide(a, b, temp, 1, CV_16S);
				jointNearestFilter(temp, destDisparityMap, Size(2 * jointNearestR + 1, 2 * jointNearestR + 1), destDisparityMap);
			}
			else if (refinementMethod == (int)REFINEMENT::WGIF_BFSUB_JNF)
			{
				weightMap.create(leftim.size(), CV_32F);
				const Size ksize = Size(2 * refinementWeightR + 1, 2 * refinementWeightR + 1);

				Mat bim;

				Mat sim; destDisparityMap.convertTo(sim, CV_32F);
				//cp::bilateralFilter(sim, bim, ksize, 100, refinementWeightR / 3.0);
				cv::bilateralFilter(sim, bim, 2 * refinementWeightR + 1, 100, refinementWeightR / 3.0);
				float* disp = sim.ptr<float>();
				float* dispb = bim.ptr<float>();
				float* s = weightMap.ptr<float>();
				for (int i = 0; i < weightMap.size().area(); i++)
				{
					float diff = (disp[i] - dispb[i]) * (disp[i] - dispb[i]) / (-2.f * 16 * 16 * refinementWeightSigma * refinementWeightSigma);
					s[i] = exp(diff);
				}

				/*
				 //min cost weight map
				uchar* minc = minCostMap.ptr<uchar>();
				float* s = weightMap.ptr<float>();
				for (int i = 0; i < weightMap.size().area(); i++)
				{
					float diff = (minc[i]) / (-2.f * 16 * 16 * refinementWeightSigma * refinementWeightSigma);
					s[i] = exp(diff);
				}*/

				Mat a, b;
				multiply(destDisparityMap, weightMap, a, 1, CV_32F);
				guidedImageFilter(a, leftim, a, refinementR, refinementSigmaRange, GUIDED_SEP_VHI);
				guidedImageFilter(weightMap, leftim, b, refinementR, refinementSigmaRange, GUIDED_SEP_VHI);
				Mat temp;
				divide(a, b, temp, 1, CV_16S);
				jointNearestFilter(temp, destDisparityMap, Size(2 * jointNearestR + 1, 2 * jointNearestR + 1), destDisparityMap);
			}
			else if (refinementMethod == (int)REFINEMENT::WGIF_BFW_JNF)
			{
				weightMap.create(leftim.size(), CV_32F);
				const Size ksize = Size(4 * refinementWeightR + 1, 4 * refinementWeightR + 1);

				cp::bilateralWeightMap(destDisparityMap, weightMap, Size(2 * rrad + 1, 2 * rrad + 1), sr1, ss, 0, 1);

				Mat a, b;
				multiply(destDisparityMap, weightMap, a, 1, CV_32F);
				guidedImageFilter(a, leftim, a, refinementR, refinementSigmaRange, GUIDED_SEP_VHI);
				guidedImageFilter(weightMap, leftim, b, refinementR, refinementSigmaRange, GUIDED_SEP_VHI);
				Mat temp;
				divide(a, b, temp, 1, CV_16S);
				jointNearestFilter(temp, destDisparityMap, Size(2 * jointNearestR + 1, 2 * jointNearestR + 1), destDisparityMap);
			}
			else if (refinementMethod == (int)REFINEMENT::WGIF_DUALBFW_JNF)
			{
				weightMap.create(leftim.size(), CV_32F);
				const Size ksize = Size(4 * refinementWeightR + 1, 4 * refinementWeightR + 1);

				cp::dualBilateralWeightMap(destDisparityMap, leftim, weightMap, Size(2 * rrad + 1, 2 * rrad + 1), sr1, sr2, 100000, 0, 1);

				Mat a, b;
				multiply(destDisparityMap, weightMap, a, 1, CV_32F);
				guidedImageFilter(a, leftim, a, refinementR, refinementSigmaRange, GUIDED_SEP_VHI);
				guidedImageFilter(weightMap, leftim, b, refinementR, refinementSigmaRange, GUIDED_SEP_VHI);
				Mat temp;
				divide(a, b, temp, 1, CV_16S);
				jointNearestFilter(temp, destDisparityMap, Size(2 * jointNearestR + 1, 2 * jointNearestR + 1), destDisparityMap);
			}
			else if (refinementMethod == (int)REFINEMENT::JBF_JNF)
			{
				Mat temp;
				jointBilateralFilter(destDisparityMap, leftim, temp, 2 * refinementR + 1, refinementSigmaRange, refinementSigmaSpace);
				jointNearestFilter(temp, destDisparityMap, Size(2 * jointNearestR + 1, 2 * jointNearestR + 1), destDisparityMap);
			}
			else if (refinementMethod == (int)REFINEMENT::WJBF_GAUSS_JNF)
			{
				Mat bim;
				GaussianBlur(destDisparityMap, bim, Size(2 * refinementWeightR + 1, 2 * refinementWeightR + 1), refinementWeightR / 3.0);
				short* disp = destDisparityMap.ptr<short>(0);
				short* dispb = bim.ptr<short>(0);
				float* s = weightMap.ptr<float>();
				for (int i = 0; i < weightMap.size().area(); i++)
				{
					float diff = (disp[i] - dispb[i]) * (disp[i] - dispb[i]) / (-2.f * 16 * 16 * refinementWeightSigma * refinementWeightSigma);
					s[i] = exp(diff);
				}

				Mat temp;
				weightedJointBilateralFilter(destDisparityMap, weightMap, leftim, temp, 2 * refinementR + 1, refinementSigmaRange, refinementSigmaSpace);
				jointNearestFilter(temp, destDisparityMap, Size(2 * jointNearestR + 1, 2 * jointNearestR + 1), destDisparityMap);
			}
			else if (refinementMethod == (int)REFINEMENT::WMF)
			{
				cp::weightedModeFilter(destDisparityMap, leftim, destDisparityMap, refinementR, refinementSigmaRange, refinementSigmaSpace, refinementSigmaHistogram);
			}
			else if (refinementMethod == (int)REFINEMENT::WWMF_GAUSS)
			{
				Mat bim;
				GaussianBlur(destDisparityMap, bim, Size(2 * refinementWeightR + 1, 2 * refinementWeightR + 1), refinementWeightR / 3.0);
				short* disp = destDisparityMap.ptr<short>(0);
				short* dispb = bim.ptr<short>(0);
				float* s = weightMap.ptr<float>();
				for (int i = 0; i < weightMap.size().area(); i++)
				{
					float diff = (disp[i] - dispb[i]) * (disp[i] - dispb[i]) / (-2.f * 16 * 16 * refinementWeightSigma * refinementWeightSigma);
					s[i] = exp(diff);
				}

				cp::weightedModeFilter(destDisparityMap, leftim, destDisparityMap, refinementR, refinementSigmaRange, refinementSigmaSpace, refinementSigmaHistogram);
			}
		}
#pragma endregion
	}

#ifdef TIMER_STEREO_MATCH
	cout << "=====================" << endl;
#endif
#pragma endregion

}

//call matching()
void StereoMatch::operator()(Mat& leftim, Mat& rightim, Mat& dest)
{
	matching(leftim, rightim, dest);
}

static void guiStereoMatchingOnMouse(int events, int x, int y, int flags, void* param)
{
	Point* pt = (Point*)param;
	//if(events==CV_EVENT_LBUTTONDOWN)
	if (flags & EVENT_FLAG_LBUTTON)
	{
		pt->x = x;
		pt->y = y;
	}
}

//call gui(Mat& leftim, Mat& rightim, Mat& destDisparity, StereoEval& eval, Mat& dmapC)
void StereoMatch::gui(Mat& leftim, Mat& rightim, Mat& destDisparity, StereoEval& eval)
{
	Mat a;
	gui(leftim, rightim, destDisparity, eval, a);
}

void StereoMatch::gui(Mat& leftim, Mat& rightim, Mat& destDisparity, StereoEval& eval, Mat& dmapC, const int offset_pointcloud)
{
	const float mask_size = 1.264f;
	const float dark_amp = 1.f / 12.f;
	const int colorSkip = true;
	vector<Mat> maskRandom(50);

	RNG rng;
	for (int i = 0; i < 50; i++)
	{
		maskRandom[i].create(leftim.size(), CV_8U);
		randu(maskRandom[i], 0, 256);
		threshold(maskRandom[i], maskRandom[i], 255 * i * 0.01, 255, THRESH_BINARY_INV);
	}

#pragma region setup
	ConsoleImage ci(Size(720, 800));
	//ci.setFontSize(12);
	string wname = "";
	string wname2 = "Disparity Map";

	namedWindow(wname2);
	moveWindow(wname2, 200, 200);

	namedWindow("LR");
	//0: proposed precomputed
	//1: random precomputed
	//2: PDS precomputed
	//3: no mask
	//4: proposed 
	//5: random
	//6: PDS
	int patternMethod = 0; createTrackbar("patternMethod", "LR", &patternMethod, 6);
	int randomSeed = 0; createTrackbar("randomSeed", "LR", &randomSeed, 99);
	int randomIndex = 2; createTrackbar("randomIdx", "LR", &randomIndex, 49);
	int display_image_depth_alpha = 0; createTrackbar("disp-image: alpha", wname2, &display_image_depth_alpha, 100);
	//pre filter
	createTrackbar("prefilter: cap", wname, &preFilterCap, 255);

	//pixelMatchingMethod = ADEdge;
	pixelMatchingMethod = CENSUS13x3;
	//pixelMatchingMethod = CENSUSTEST;

	createTrackbar("pix match: method", wname, &pixelMatchingMethod, Pixel_Matching_Method_Size - 1);
	createTrackbar("pix match: color method", wname, &color_distance, ColorDistance_Size - 1);
	createTrackbar("pix match: blend a", wname, &costAlphaImageSobel, 100);
	pixelMatchErrorCap = preFilterCap;
	createTrackbar("pix match: err cap", wname, &pixelMatchErrorCap, 255);

	//AggregationMethod = Aggregation_Gauss;
	//aggregationMethod = Guided;
	aggregationMethod = Box;
	createTrackbar("agg: method", wname, &aggregationMethod, Aggregation_Method_Size - 1);
	aggregationRadiusH = 4; createTrackbar("agg: r width", wname, &aggregationRadiusH, 20);
	aggregationRadiusV = 4; createTrackbar("agg: r height", wname, &aggregationRadiusV, 20);
	int aggeps = 1; createTrackbar("agg: guide color sigma/eps", wname, &aggeps, 255);
	int aggss = 100; createTrackbar("agg: guide space sigma", wname, &aggss, 255);

	// disable P1, P2 for optimization(under debug)
	//createTrackbar("P1", wname, &P1, 20);
	//createTrackbar("P2", wname, &P2, 20);

	uniquenessRatio = 10;
	createTrackbar("uniq", wname, &uniquenessRatio, 100);
	createTrackbar("subpixel RF widow size", wname, &subpixelRangeFilterWindow, 10);
	createTrackbar("subpixel RF cap", wname, &subpixelRangeFilterCap, 128);

	createTrackbar("LR check disp12", wname, &disp12diff, 100);

	//int spsize = 300;
	speckleWindowSize = 20;
	createTrackbar("speckleSize", wname, &speckleWindowSize, 1000);
	speckleRange = 16;
	createTrackbar("speckleDiff", wname, &speckleRange, 100);

	createTrackbar("occlusionMethod", wname, &holeFillingMethod, 1);

	Plot costFunctionPlot(Size(640, 240));
	Plot signalPlot(Size(640, 240));
	int vh = 0;
	int diffmode = 0;
	if (eval.isInit)
	{
		namedWindow("signal");
		createTrackbar("vh", "signal", &vh, 1);
		createTrackbar("mode", "signal", &diffmode, 2);
	}

	Point mpt = Point(100, 100);
	createTrackbar("px", wname, &mpt.x, leftim.cols - 1);
	createTrackbar("py", wname, &mpt.y, leftim.rows - 1);

	bool isStreak = false;
	bool isMedian = false;

	int maskType = 1;//0: nomask, 1: nonocc: 2, all: 3, disc, 4: GT
	int maskPrec = 2;//0: 0.5, 1: 1.0, 2: 2.0
	bool isPlotCostFunction = true;
	bool isPlotSignal = true;
	bool isGrid = true;
	bool isDispalityColor = false;

	setMouseCallback(wname2, guiStereoMatchingOnMouse, &mpt);
	//CostVolumeRefinement cbf(minDisparity, numberOfDisparities);

	Mat dispShow;
	destDisparity.setTo(0);

	//for projection
	Mat L = leftim.clone();
	Mat R = rightim.clone();
#pragma endregion
	int key = 0;
#ifdef GUI_LOOP
	while (key != 'q') {
#else
	for (int n = 0; n < 6; n++)
	{
		if (n == 0)pixelMatchingMethod = CENSUS13x3;
		if (n == 1)pixelMatchingMethod = CENSUS7x3;
		if (n == 2)pixelMatchingMethod = AD;
		if (n == 3)pixelMatchingMethod = SD;
		if (n == 4)pixelMatchingMethod = ADEdge;
		if (n == 5)pixelMatchingMethod = SDEdge;
#endif
#pragma region parameter setup
		//init
		aggregationGuidedfilterEps = aggeps;
		aggregationSigmaSpace = aggss;
#pragma endregion

#pragma region console setup

		if (pixelMatchingMethod % 2 == 0)
			ci("Cost            (i-u)| " + getCostMethodName((Cost)pixelMatchingMethod) + "");
		else
			ci("Cost      (i-u)|(j-k)| " + getCostMethodName((Cost)pixelMatchingMethod) + getCostColorDistanceName((ColorDistance)color_distance));
		ci("Aggregation     (@-[)| " + getAggregationMethodName((Aggregation)aggregationMethod));

		if (isUniquenessFilter) ci(CV_RGB(0, 255, 0), "uniqueness filter (1)| true");
		else ci(CV_RGB(255, 0, 0), "uniqueness filter (1)| false");

		if (subpixelInterpolationMethod == 0)ci(CV_RGB(255, 0, 0), "subpixel          (2)| NONE");
		else ci(CV_RGB(0, 255, 0), "subpixel          (2)| " + getSubpixelInterpolationMethodName((SUBPIXEL)subpixelInterpolationMethod));

		if (isRangeFilterSubpix) ci(CV_RGB(0, 255, 0), "Range filter      (3)| true");
		else ci(CV_RGB(255, 0, 0), "range filter      (3)| false");

		if (LRCheckMethod) ci(CV_RGB(0, 255, 0), "LR check          (4)| " + getLRCheckMethodName((LRCHECK)LRCheckMethod));
		else ci(CV_RGB(255, 0, 0), "LR check          (4)| false");

		if (isProcessLBorder) ci(CV_RGB(0, 255, 0), "LBorder           (5)| LR check must be true: true");
		else ci(CV_RGB(255, 0, 0), "LBorder           (5)| false");

		if (isMinCostFilter) ci(CV_RGB(0, 255, 0), "min cost filter   (6)| true");
		else ci(CV_RGB(255, 0, 0), "min cost filter   (6)| false");

		if (isSpeckleFilter) ci(CV_RGB(0, 255, 0), "speckle filter    (7)| true");
		else ci(CV_RGB(255, 0, 0), "speckle filter    (7)| false");

		if (holeFillingMethod == 0) ci(CV_RGB(255, 0, 0), "Occlusion         (8)| NONE");
		else ci(CV_RGB(0, 255, 0), "Occlusion         (8)| " + getHollFillingMethodName((HOLE_FILL)holeFillingMethod));


		ci("==== additional post filter =====");
		if (isStreak)ci(CV_RGB(0, 255, 0), "Streak            (0)| true");
		else ci(CV_RGB(255, 0, 0), "Streak            (0)| false");

		if (isMedian)ci(CV_RGB(0, 255, 0), "Median            (-)| true");
		else ci(CV_RGB(255, 0, 0), "Median            (-)| false");

		ci("=======================");
		if (maskType != 0)
		{
			if (maskType == 4)
				ci(CV_RGB(255, 0, 0), "mask (m-,): ground trueth");

			if (maskPrec == 2 && maskType == 1)
				ci(CV_RGB(0, 255, 0), "mask (m-,): nonocc, prec: 2.0");
			if (maskPrec == 1 && maskType == 1)
				ci(CV_RGB(0, 255, 0), "mask (m-,): nonocc, prec: 1.0");
			if (maskPrec == 0 && maskType == 1)
				ci(CV_RGB(0, 255, 0), "mask (m-,): nonocc, prec: 0.5");

			if (maskPrec == 2 && maskType == 2)
				ci(CV_RGB(0, 255, 0), "mask (m-,): all, prec: 2.0");
			if (maskPrec == 1 && maskType == 2)
				ci(CV_RGB(0, 255, 0), "mask (m-,): all, prec: 1.0");
			if (maskPrec == 0 && maskType == 2)
				ci(CV_RGB(0, 255, 0), "mask (m-,): all, prec: 0.5");

			if (maskPrec == 2 && maskType == 3)
				ci(CV_RGB(0, 255, 0), "mask (m-,): disc, prec: 2.0");
			if (maskPrec == 1 && maskType == 3)
				ci(CV_RGB(0, 255, 0), "mask (m-,): disc, prec: 1.0");
			if (maskPrec == 0 && maskType == 3)
				ci(CV_RGB(0, 255, 0), "mask (m-,): disc, prec: 0.5");
		}
		else
		{
			ci(CV_RGB(255, 0, 0), "mask (m-,): none");
		}

#pragma endregion

#pragma region projectPattern
		{
			Mat mask(leftim.size(), CV_8U);
			if (patternMethod == 0) mask = imread("maskProp.png", 0);
			else if (patternMethod == 1)mask = imread("maskRand.png", 0);
			else if (patternMethod == 2)mask = imread("maskPDS.png", 0);
			else if (patternMethod == 3)mask.setTo(0);
			else if (patternMethod == 4|| patternMethod == 6)
			{
				loadPrecomputedPDS(mask, randomIndex % 10, randomSeed);
				if(patternMethod == 4)addRandmizedSatellitePoints(mask, mask, rng);
				//randmizedSatellitePoints(mask, mask, rng);
			}
			else if (patternMethod == 5)
			{
				mask = maskRandom[randomIndex].clone();
				//randu(mask, 0, 256);
				//threshold(mask, mask, 255 * randomIndex * 0.01, 255, THRESH_BINARY_INV);
			}

			const float ratio = float(countNonZero(mask)) / leftim.size().area();

			//Timer tt("projection");
			//leftim.copyTo(L); rightim.copyTo(R);// noprojection
			projectPattern(leftim, L, rightim, R, dmapC, mask, mask_size, dark_amp);

			//Mat temp;  cvtColor(convertGray(L, CV_8U), temp, COLOR_GRAY2BGR);
			//imwrite("L.png", temp);
			//cvtColor(convertGray(R, CV_8U), temp, COLOR_GRAY2BGR);
			//imwrite("R.png", temp);
			//guiCropZoom(temp);

			Mat LR; hconcat(L, R, LR);
			string mes = "";
			switch (patternMethod)
			{
			case 0: mes = "proposed precomputed"; break;
			case 1: mes = "random precomputed"; break;
			case 2: mes = "PDS precomputed"; break;
			case 3: mes = "no mask"; break;
			case 4: mes = "proposed "; break;
			case 5: mes = "random"; break;
			case 6: mes = "PDS"; break;
			default:
				break;
			}
			//format("ratio %6.2f", ratio * 100.0)
			cv::addText(LR, mes, Point(30, 60), "Arial", 60, COLOR_WHITE);
			cv::addText(LR, format("ratio %6.2f", ratio * 100.0), Point(30, 150), "Arial", 60, COLOR_WHITE);
			imshowResize("LR", LR, Size(), 0.25, 0.25);
		}
#pragma endregion

#pragma region stereo body
		{
			Timer t("BM", TIME_MSEC, false);
			matching(L, R, destDisparity);
			ci("Time StereoMatch | %6.2f ms", t.getTime());
		}

		//additional post process
		{
			Mat base = destDisparity.clone();
#ifdef TIMER_STEREO_MATCH
			Timer t("Post 2", TIME_MSEC);
#else 
			Timer t("Post 2", TIME_MSEC, false);
#endif
			if (isStreak)
			{
				removeStreakingNoise(destDisparity, destDisparity, 16);
				removeStreakingNoiseV(destDisparity, destDisparity, 16);
			}
			if (isMedian)
			{
				medianBlur(destDisparity, destDisparity, 3);
			}
			ci("Time A-PostFilter| %6.2f ms", t.getTime());//additional post processing time

		}
#pragma endregion 

#pragma region show
		ci("valid (v)        | %5.2f %%", getValidRatio());

		//plot cost function
		if (isPlotCostFunction)
		{
			costFunctionPlot.clear();
			costFunctionPlot.setYLabel("error");
			costFunctionPlot.setXLabel("disparity");
			costFunctionPlot.setPlotTitle(0, "cost");
			costFunctionPlot.setPlotTitle(1, "answer");
			costFunctionPlot.setPlotTitle(2, "result");
			costFunctionPlot.setGrid(0);
			costFunctionPlot.setIsDrawMousePosition(false);
			costFunctionPlot.setXYRange(0, numberOfDisparities + minDisparity + 1, 0, 64);

			if (eval.isInit)
			{
				const int dd = (int)(eval.ground_truth.at<uchar>(mpt.y, mpt.x) / eval.amp + 0.5);
				const int dd2 = (int)((double)destDisparity.at<short>(mpt.y, mpt.x) / (16.0) + 0.5);

				const double ddd = (eval.ground_truth.at<uchar>(mpt.y, mpt.x) / eval.amp);
				const double ddd2 = ((double)destDisparity.at<short>(mpt.y, mpt.x) / (16.0));
				for (int i = 0; i < numberOfDisparities; i++)
				{
					costFunctionPlot.push_back(i + minDisparity, DSI[i].at<uchar>(mpt.y, mpt.x), 0);
				}
				costFunctionPlot.push_back_VLine(ddd, 1);
				costFunctionPlot.push_back_VLine(ddd2, 2);
			}
			else
			{
				const int dd2 = (int)((double)destDisparity.at<short>(mpt.y, mpt.x) / (16.0) + 0.5);
				const double ddd2 = ((double)destDisparity.at<short>(mpt.y, mpt.x) / (16.0));
				for (int i = 0; i < numberOfDisparities; i++)
				{
					costFunctionPlot.push_back(i + minDisparity, DSI[i].at<uchar>(mpt.y, mpt.x), 0);

					if (abs(i + minDisparity - dd2) == 0)
						costFunctionPlot.push_back(ddd2, 0, 2);
					else
						costFunctionPlot.push_back(i + minDisparity, 64, 2);
				}
			}

			costFunctionPlot.plot("cost function", false);

		}

		//plot signal
		if (isPlotSignal)
		{
			signalPlot.clear();
			signalPlot.setXLabel("position");
			signalPlot.setYLabel("disparity");
			signalPlot.setPlotTitle(0, "estimate");
			signalPlot.setPlotTitle(1, "ground trouth");
			signalPlot.setGrid(0);
			signalPlot.setIsDrawMousePosition(false);
			const int mindisp = 15;
			if (vh == 0)
			{
				signalPlot.setXYRange(0, destDisparity.cols - 1, mindisp, numberOfDisparities);
				signalPlot.setPlot(0, CV_RGB(0, 0, 0), 0, 1, 1);
				signalPlot.setPlot(1, CV_RGB(255, 0, 0), 0, 1, 1);
				for (int i = 0; i < destDisparity.cols; i++)
				{
					double ddd = (eval.ground_truth.at<uchar>(mpt.y, i) / eval.amp);
					double ddd2 = ((double)destDisparity.at<short>(mpt.y, i) / (16.0));

					if (diffmode == 1)
					{
						ddd = abs(ddd - (eval.ground_truth.at<uchar>(mpt.y, i - 1) / eval.amp));
						ddd2 = abs(ddd2 - (((double)destDisparity.at<short>(mpt.y, i - 1) / (16.0))));
					}
					if (diffmode == 2)
					{
						ddd = abs((ddd - (eval.ground_truth.at<uchar>(mpt.y, i - 1) / eval.amp)) - (ddd - (eval.ground_truth.at<uchar>(mpt.y, i + 1) / eval.amp)));
						ddd2 = abs((ddd2 - ((double)destDisparity.at<short>(mpt.y, i - 1) / (16.0))) - (ddd2 - ((double)destDisparity.at<short>(mpt.y, i + 1) / (16.0))));
					}

					signalPlot.push_back(i, ddd2, 0);
					signalPlot.push_back(i, ddd, 1);
				}
				signalPlot.plot("signal", false);
			}
			else
			{
				signalPlot.setXYRange(0, destDisparity.rows - 1, mindisp, numberOfDisparities);
				signalPlot.setPlot(0, CV_RGB(0, 0, 0), 0, 1, 1);
				signalPlot.setPlot(1, CV_RGB(255, 0, 0), 0, 1, 1);
				for (int i = 0; i < destDisparity.rows; i++)
				{
					double ddd = (eval.ground_truth.at<uchar>(i, mpt.x) / eval.amp);
					double ddd2 = ((double)destDisparity.at<short>(i, mpt.x) / (16.0));

					if (diffmode == 1)
					{
						ddd = abs(ddd - (eval.ground_truth.at<uchar>(i - 1, mpt.x) / eval.amp));
						ddd2 = abs(ddd2 - (((double)destDisparity.at<short>(i - 1, mpt.x) / (16.0))));
					}
					if (diffmode == 2)
					{
						ddd = abs(-(ddd - (eval.ground_truth.at<uchar>(i - 1, mpt.x) / eval.amp)) + (-ddd + (eval.ground_truth.at<uchar>(i + 1, mpt.x) / eval.amp)));
						ddd2 = abs(-(ddd2 - ((double)destDisparity.at<short>(i - 1, mpt.x) / (16.0))) + (-ddd2 + ((double)destDisparity.at<short>(i + 1, mpt.x) / (16.0))));
					}

					signalPlot.push_back(i, ddd2, 0);
					signalPlot.push_back(i, ddd, 1);
				}
				signalPlot.plotData();
				Mat show;
				transpose(signalPlot.render, show);
				flip(show, show, 1);
				imshow("signal", show);
			}
		}

		//compute StereoEval
		if (eval.isInit)
		{
			Mat maskbadpixel = Mat::zeros(destDisparity.size(), CV_8U);
			bool isPrintEval = false;
			ci("Th |noocc,   all,  disc");
			//ci("0.5|" + eval(destDisparity, 0.5, 16, isPrintEval));
			ci("1.0|" + eval(destDisparity, 1.0, 16, isPrintEval));
			//imshow("th=1", eval.nonocc_th);
			//imshow("th=1", eval.all_th);
			//ci("2.0|" + eval(destDisparity, 2.0, 16, isPrintEval));
			//ci("MSE|" + eval.getMSE(destDisparity, 16, isPrintEval));
			csv.write(eval.nonocc);
			//csv.write(eval.nonoccMSE);
			if (maskType != 0)
			{
				if (maskPrec == 0)eval(destDisparity, 0.5, 16, isPrintEval);
				if (maskPrec == 1)eval(destDisparity, 1.0, 16, isPrintEval);
				if (maskPrec == 2)eval(destDisparity, 2.0, 16, isPrintEval);

				if (maskType == 1)
					eval.nonocc_th.copyTo(maskbadpixel);
				else if (maskType == 2)
					eval.all_th.copyTo(maskbadpixel);
				else if (maskType == 3)
					eval.disc_th.copyTo(maskbadpixel);
			}
			if (maskType == 4)
			{
				if (isDispalityColor) { Mat a; eval.ground_truth.convertTo(a, CV_16S, 4); cvtDisparityColor(a, dispShow, minDisparity, numberOfDisparities - 10, DISPARITY_COLOR::COLOR_PSEUDO, 16); }
				else eval.ground_truth.copyTo(dispShow);
			}
			else
			{
				dispShow.setTo(Scalar(0, 0, 255), maskbadpixel);
			}
		}

		// show disparity
		if (eval.isInit)
		{
			if (isDispalityColor)cvtDisparityColor(destDisparity, dispShow, minDisparity, numberOfDisparities, DISPARITY_COLOR::COLOR_PSEUDO, 16);
			else				 cvtDisparityColor(destDisparity, dispShow, 0, 255, DISPARITY_COLOR::GRAY, int(16 / eval.amp));
		}
		else
		{
			if (isDispalityColor)cvtDisparityColor(destDisparity, dispShow, minDisparity, numberOfDisparities, DISPARITY_COLOR::COLOR_PSEUDO, 16);
			else				 cvtDisparityColor(destDisparity, dispShow, 0, numberOfDisparities, DISPARITY_COLOR::GRAY);
		}
		alphaBlend(leftim, dispShow, display_image_depth_alpha / 100.0, dispShow);
		if (isGrid)
		{
			line(dispShow, Point(0, mpt.y), Point(leftim.cols, mpt.y), CV_RGB(0, 255, 0));
			line(dispShow, Point(mpt.x, 0), Point(mpt.x, leftim.rows), CV_RGB(0, 255, 0));
		}

		imshow(wname2, dispShow);

		ci("Other keys");
		ci("grid (g), dispcolor(w),");
		ci("check alpha(c), point cloud (p)");

		moveWindow("console", 1500, 300);
		ci.show();
		setTrackbarPos("px", wname, mpt.x);
		setTrackbarPos("py", wname, mpt.y);
#pragma endregion 

#pragma region key input
		key = waitKey(1);
		if (key == '1') isUniquenessFilter = isUniquenessFilter ? false : true;
		if (key == '2') { subpixelInterpolationMethod++; subpixelInterpolationMethod = (subpixelInterpolationMethod > 2) ? 0 : subpixelInterpolationMethod; }
		if (key == '3')	isRangeFilterSubpix = (isRangeFilterSubpix) ? false : true;
		if (key == '4') { LRCheckMethod++;  LRCheckMethod = (LRCheckMethod > (int)LRCHECK::LRCHECK_SIZE - 1) ? 0 : LRCheckMethod; }
		if (key == '5') isProcessLBorder = (isProcessLBorder) ? false : true;
		if (key == '6') isMinCostFilter = isMinCostFilter ? false : true;
		if (key == '7') isSpeckleFilter = isSpeckleFilter ? false : true;
		if (key == '8') { holeFillingMethod++; if (holeFillingMethod > 1)holeFillingMethod = 0; }
		if (key == 'v')  holeFillingMethod = (holeFillingMethod != 0) ? 0 : 1;
		if (key == '0') isStreak = (isStreak) ? false : true;
		if (key == '-') isMedian = (isMedian) ? false : true;

		if (key == 'i')
		{
			if (colorSkip)
			{
				pixelMatchingMethod += 2; pixelMatchingMethod = (pixelMatchingMethod > Pixel_Matching_Method_Size - 1) ? 0 : pixelMatchingMethod;
			}
			else
			{
				pixelMatchingMethod++; pixelMatchingMethod = (pixelMatchingMethod > Pixel_Matching_Method_Size - 1) ? 0 : pixelMatchingMethod;
			}
		}
		if (key == 'u')
		{
			if (colorSkip)
			{
				pixelMatchingMethod -= 2; pixelMatchingMethod = (pixelMatchingMethod < 0) ? Pixel_Matching_Method_Size - 2 : pixelMatchingMethod;
			}
			else
			{
				pixelMatchingMethod--; pixelMatchingMethod = (pixelMatchingMethod < 0) ? Pixel_Matching_Method_Size - 2 : pixelMatchingMethod;
			}
		}

		if (key == 'j') { color_distance++; color_distance = (color_distance > ColorDistance_Size - 1) ? 0 : color_distance; }
		if (key == 'k') { color_distance--; color_distance = (color_distance < 0) ? ColorDistance_Size - 2 : color_distance; }

		if (key == '@') { aggregationMethod++; aggregationMethod = (aggregationMethod > Aggregation_Method_Size - 1) ? 0 : aggregationMethod; }
		if (key == '[') { aggregationMethod--; aggregationMethod = (aggregationMethod < 0) ? Aggregation_Method_Size - 2 : aggregationMethod; }

		if (key == 'm')maskType++; maskType = maskType > 4 ? 0 : maskType;
		if (key == ',')maskPrec++; maskPrec = maskPrec > 2 ? 0 : maskPrec;

		if (key == 'g') isGrid = (isGrid) ? false : true;
		if (key == 'w')isDispalityColor = (isDispalityColor) ? false : true;
		if (key == 'c') guiAlphaBlend(dispShow, leftim);
		if (key == 'p')
		{
			cp::PointCloudShow pcs;
			//The focal length is 3740 pixels, and the baseline is 160mm (Middlebury 2005,2006);
			//Middlebury 2005,2006 require offset value for disparity maps denoted in each file ;
			add(destDisparity, offset_pointcloud * 16, destDisparity);
			pcs.loop(leftim, destDisparity, 16.f, 3740.f, 160.f * 0.1, 0);
		}
#pragma endregion
}
}

void StereoMatch::showWeightMap(std::string wname)
{
	if (!weightMap.empty())
	{
		cp::imshowNormalize(wname, weightMap);
	}
}

#pragma endregion

#pragma region prefilter

static void prefilterXSobel(Mat & src, Mat & dst, const int preFilterCap)
{
	if (dst.empty() || dst.depth() != CV_8U)dst.create(src.size(), CV_8U);

	const int OFS = 256 * 4, TABSZ = OFS * 2 + 256;

	Size size = src.size();
	const uchar val0 = preFilterCap;
	const int preFilterCap2 = saturate_cast<int>(2 * preFilterCap);

	const int step = 2 * size.width;//2 * size.width
	uchar* srow1 = src.ptr<uchar>();
	uchar* dptr0 = dst.ptr<uchar>();
	uchar* dptr1 = dptr0 + dst.step;

	const int WIDTH = get_simd_floor(size.width - 1, 8);
	const int e = size.width - 1;
	const int WCS = 2;//center weight of Sobel
	int y;
	//unrolling y0 and y1
	const int HEIGHT = get_simd_floor(size.height, 2);
	for (y = 0; y < HEIGHT; y += 2)
	{
		const uchar* srow0 = y > 0 ? srow1 - src.step : size.height > 1 ? srow1 + src.step : srow1;
		const uchar* srow2 = y < size.height - 1 ? srow1 + src.step : size.height > 1 ? srow1 - src.step : srow1;
		const uchar* srow3 = y < size.height - 2 ? srow1 + src.step * 2 : srow1;

		dptr0[0] = saturate_cast<uchar>(min(preFilterCap2, (srow0[0] - srow0[1]) + WCS * (srow1[0] - srow1[1]) + (srow2[0] - srow2[1]) + preFilterCap));
		dptr1[0] = saturate_cast<uchar>(min(preFilterCap2, (srow1[0] - srow1[1]) + WCS * (srow2[0] - srow2[1]) + (srow3[0] - srow3[1]) + preFilterCap));

		__m128i zero = _mm_setzero_si128();
		__m128i ftz = _mm_set1_epi16((short)preFilterCap); //preFilterCap (short)
		__m128i ftz2 = _mm_set1_epi8(saturate_cast<uchar>(preFilterCap2));//preFilterCap2 (uchar)
#if 1
		for (int x = 1; x < WIDTH; x += 8)
		{
			__m128i c0 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow0 + x - 1)), zero);
			__m128i c1 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow1 + x - 1)), zero);
			__m128i d0 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow0 + x + 1)), zero);
			__m128i d1 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow1 + x + 1)), zero);

			d0 = _mm_sub_epi16(d0, c0);
			d1 = _mm_sub_epi16(d1, c1);

			__m128i c2 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow2 + x - 1)), zero);
			__m128i c3 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow2 + x - 1)), zero);
			__m128i d2 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow3 + x + 1)), zero);
			__m128i d3 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(srow3 + x + 1)), zero);

			d2 = _mm_sub_epi16(d2, c2);
			d3 = _mm_sub_epi16(d3, c3);

			__m128i v0 = _mm_add_epi16(d0, _mm_add_epi16(d2, _mm_add_epi16(d1, d1)));
			__m128i v1 = _mm_add_epi16(d1, _mm_add_epi16(d3, _mm_add_epi16(d2, d2)));
			v0 = _mm_packus_epi16(_mm_add_epi16(v0, ftz), _mm_add_epi16(v1, ftz));
			v0 = _mm_min_epu8(v0, ftz2);

			_mm_storel_epi64((__m128i*)(dptr0 + x), v0);
			_mm_storel_epi64((__m128i*)(dptr1 + x), _mm_unpackhi_epi64(v0, v0));
		}
		for (int x = WIDTH; x < size.width - 1; x++)
		{
			dptr0[x] = saturate_cast<uchar>(min(preFilterCap2, (srow0[x + 1] - srow0[x - 1]) + WCS * (srow1[x + 1] - srow1[x - 1]) + (srow2[x + 1] - srow2[x - 1]) + preFilterCap));
			dptr1[x] = saturate_cast<uchar>(min(preFilterCap2, (srow1[x + 1] - srow1[x - 1]) + WCS * (srow2[x + 1] - srow2[x - 1]) + (srow3[x + 1] - srow3[x - 1]) + preFilterCap));
		}
#else
		for (int x = 1; x < size.width - 1; x++)
		{
			dptr0[x] = saturate_cast<uchar>(min(preFilterCap2, (srow0[x + 1] - srow0[x - 1]) + WCS * (srow1[x + 1] - srow1[x - 1]) + (srow2[x + 1] - srow2[x - 1]) + preFilterCap));
			dptr1[x] = saturate_cast<uchar>(min(preFilterCap2, (srow1[x + 1] - srow1[x - 1]) + WCS * (srow2[x + 1] - srow2[x - 1]) + (srow3[x + 1] - srow3[x - 1]) + preFilterCap));
		}
#endif
		dptr0[e] = saturate_cast<uchar>(min(preFilterCap2, (srow0[e] - srow0[e - 1]) + WCS * (srow1[e] - srow1[e - 1]) + (srow2[e] - srow2[e - 1]) + preFilterCap));
		dptr1[e] = saturate_cast<uchar>(min(preFilterCap2, (srow1[e] - srow1[e - 1]) + WCS * (srow2[e] - srow2[e - 1]) + (srow3[e] - srow3[e - 1]) + preFilterCap));
		srow1 += step;
		dptr0 += step;
		dptr1 += step;
		}
	srow1 -= src.cols;

	for (int y = HEIGHT; y < size.height; y++)
	{
		uchar* dptr = dst.ptr<uchar>(y);
		const uchar* srow0 = srow1 - src.step;
		dptr[0] = saturate_cast<uchar>(min(preFilterCap2, 2 * (srow0[0] - srow0[1]) + WCS * (srow1[0] - srow1[1]) + preFilterCap));
		for (int x = 1; x < size.width - 1; x++)
		{
			dptr[x] = saturate_cast<uchar>(min(preFilterCap2, 2 * (srow0[x - 1] - srow0[x + 1]) + WCS * (srow1[x - 1] - srow1[x + 1]) + preFilterCap));
		}
		dptr[e] = saturate_cast<uchar>(min(preFilterCap2, 2 * (srow0[e - 1] - srow0[e]) + WCS * (srow1[e - 1] - srow1[e]) + preFilterCap));
	}
	}

static void prefilterGuided(Mat & src, Mat & dst, const int preFilterCap)
{
	dst.create(src.size(), CV_8U);
	Mat dest(src.size(), CV_32F);
	guidedImageFilter(src, src, dest, 2, 70, GUIDED_SEP_VHI);
	uchar* s = src.ptr<uchar>();
	float* d0 = dest.ptr<float>();
	uchar* d1 = dst.ptr<uchar>();
	for (int i = 0; i < src.size().area(); i++)
	{
		d1[i] = saturate_cast<uchar>(min(2.f * preFilterCap, preFilterCap - 5.f * (s[i] - d0[i])));
	}
	imshow("a", dst);
}

static void censusTrans8U_3x3(Mat & src, Mat & dest)
{
	if (dest.empty() || dest.depth() != CV_8U)dest.create(src.size(), CV_8U);
	const int r = 1;
	const int r2 = 2 * r;
	Mat im; copyMakeBorder(src, im, r, r, r, r, cv::BORDER_REFLECT101);

	uchar* s = im.ptr<uchar>(r); s += r;
	uchar* d = dest.ptr<uchar>(0);
	uchar* sb;

	const int step1 = -r - im.cols;
	const int step2 = -3 + im.cols;
	const int w = src.cols;
	const int h = src.rows;
	for (int j = 0; j < h; j++)
	{
		for (int i = 0; i < w; i++)
		{
			uchar val = 0;//init value
			sb = s + step1;

			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			sb += step2;

			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			sb++;//skip r=0
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			sb += step2;

			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			val = (*sb < *s) ? val | 1 : val; sb++;

			*(d++) = val;
			s++;
		}
		s += r2;
	}
}

static void censusTrans32S_5x5(Mat & src, Mat & dest)
{
	if (dest.empty() || dest.depth() != CV_32S)dest.create(src.size(), CV_32S);
	const int r = 2;
	const int r2 = 2 * r;
	const int D = 2 * r + 1;
	Mat im; copyMakeBorder(src, im, r, r, r, r, cv::BORDER_REFLECT101);

	uchar* sb;//around
	uchar* s = im.ptr<uchar>(r); s += r;
	int* d = dest.ptr<int>();
	const int step1 = -r - r * im.cols;
	const int step2 = -D + im.cols;
	const int w = src.cols;
	const int h = src.rows;
	for (int j = 0; j < h; j++)
	{
		for (int i = 0; i < w; i++)
		{
			int val = 0;//init value
			sb = s + step1;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;

			sb += step2;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;

			sb += step2;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			sb++;//skip r=0
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;

			sb += step2;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;

			sb += step2;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			val = (*sb < *s) ? val | 1 : val; sb++;

			*(d++) = val;
			s++;
		}
		s += r2;
	}
}

// xxxxx 
//xxxxxxx
//xxx xxx
//xxxxxxx
// xxxxx
static void censusTrans32S_7x5_30(Mat & src, Mat & dest)
{
	if (dest.empty() || dest.depth() != CV_32S)dest.create(src.size(), CV_32S);
	const int r = 3;
	const int r2 = 2 * r;
	const int D = 2 * r + 1;
	const int vr = 2;
	Mat im; copyMakeBorder(src, im, vr, vr, r, r, cv::BORDER_REFLECT101);

	uchar* s = im.ptr<uchar>(vr); s += r;
	uchar* sb;
	int* d = dest.ptr<int>();

	const int step1 = -r - vr * im.cols;
	const int step2 = -D + im.cols;
	const int w = src.cols;
	const int h = src.rows;
	for (int j = 0; j < h; j++)
	{
		for (int i = 0; i < w; i++)
		{
			int val = 0;//init value
			sb = s + step1;
			sb++;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			sb++;

			sb += step2;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;

			sb += step2;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			sb++;//skip r=0
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;

			sb += step2;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;

			sb += step2;
			sb++;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			val = (*sb < *s) ? val | 1 : val; sb++; val <<= 1;
			val = (*sb < *s) ? val | 1 : val; sb++;
			sb++;

			*(d++) = val;
			s++;
		}
		s += r2;
	}
}

void generateCensusIndex(const Mat & mask, const int borderCols, int* cindex)
{
	const uchar* m = mask.ptr<const uchar>();
	int idx = 0;
	for (int j = 0; j < mask.rows; j++)
	{
		for (int i = 0; i < mask.cols; i++)
		{
			if (*m++ != 0)
			{
				cindex[idx++] = borderCols * j + i;
			}
		}
	}
}

int localCensusTransform(const uchar * src, const uchar scenter, int* cindex, const int csize)
{
	int val = 0;
	for (int i = 0; i < csize; i++)
	{
		val = (src[cindex[i]] < scenter) ? val | 1 : val;
		val <<= 1;
	}
	return val;
}

int localCensusTransformUsingMask(const uchar * src, const Mat & mask, const int step1, const int step2)
{
	int val = 0;
	const uchar* sb = src + step1;
	const uchar* m = mask.ptr<const uchar>();
	for (int j = 0; j < mask.rows; j++)
	{
		for (int i = 0; i < mask.cols; i++)
		{
			if (*m++ != 0)
			{
				val = (*sb < *src) ? val | 1 : val;
				val <<= 1;
			}
			sb++;
		}
		sb += step2;
	}
	return val;
}

int localSymmetricCensusTransform(const uchar * src, const Mat & mask, const int step1, const int step2)
{
	int val = 0;
	const uchar* sb = src + step1;
	const uchar* sr = src - step1;
	const uchar* m = mask.ptr<const uchar>();
	const int h = mask.rows >> 1;
	const int w = mask.cols >> 1;
	for (int j = 0; j < h; j++)
	{
		for (int i = 0; i < mask.cols; i++)
		{
			if (*m++ != 0)
			{
				val = (*sb < *sr) ? val | 1 : val;
				val <<= 1;
				//val = (*sb < *sr) ? val | 1 : val;val <<= 1;
			}
			sb++;
			sr--;
		}
		sb += step2;
		sr -= step2;
	}
	//last
	{
		for (int i = 0; i < w; i++)
		{
			if (*m++ != 0)
			{
				val = (*sb < *sr) ? val | 1 : val;
				val <<= 1;
				//val = (*sb < *sr) ? val | 1 : val;val <<= 1;
			}
			sb++;
			sr--;
		}
	}
	return val;
}

static void censusTransform32S(Mat & src, Mat & dest, Mat & censusMask, const bool isSymmetric = false)
{
	if (dest.empty() || dest.depth() != CV_32S)dest.create(src.size(), CV_32S);
	const int r = censusMask.cols >> 1;
	const int r2 = 2 * r;
	const int D = 2 * r + 1;
	const int vr = censusMask.rows >> 1;
	Mat im; copyMakeBorder(src, im, vr, vr, r, r, cv::BORDER_REFLECT101);

	const int step1 = -r - vr * im.cols;
	const int step2 = -D + im.cols;
	const int w = src.cols;
	const int h = src.rows;
	const int csize = countNonZero(censusMask);
	AutoBuffer<int> cindex(csize);
	generateCensusIndex(censusMask, im.cols, cindex);

#pragma omp parallel for
	for (int j = 0; j < h; j++)
	{
		//uchar* s = im.ptr<uchar>(vr + j, r);for UsingMask
		uchar* s = im.ptr<uchar>(j);
		int* d = dest.ptr<int>(j);
		for (int i = 0; i < w; i++)
		{
			*(d++) = localCensusTransform(s, s[-step1], cindex, csize);
			//*(d++) = localCensusTransformUsingMask(s, censusMask, step1, step2);// before optimization
			//*(d++) = localSymmetricCensusTransform(s, census, step1, step2);
			//int val = localCensusTransform(s, census, step1, step2) << 24;
			//*(d++) = val | localSymmetricCensusTransform(s, census, step1, step2);
			s++;
		}
		s += r2;
	}
}

// 0 1 2 3 4 5 6 7 8 9101112
// x x       x x x       x x
//           x   x       
// x x       x x x       x x
static void censusTrans32S_13x3_16(Mat & src, Mat & dest)
{
	uchar data[39] =
	{
		1,1,0,0,0,1,1,1,0,0,0,1,1,
		0,0,0,0,0,1,0,1,0,0,0,0,0,
		1,1,0,0,0,1,1,1,0,0,0,1,1
	};
	Mat census(3, 13, CV_8U, data);
	censusTransform32S(src, dest, census);

	/*uchar data[21] =
	{
		0,1,1,1,1,1,0,
		1,1,1,0,1,1,1,
		0,1,1,1,1,1,0
	};
	Mat census(3, 7, CV_8U, data);
	censusTransform32S(src, dest, census);*/
}

// 0 1 2 3 4 5 6 
//   x x x x x  
// x x x   x x x     
//   x x x x x  
static void censusTrans32S_7x3_16(Mat & src, Mat & dest)
{
	uchar data[21] =
	{
		0,1,1,1,1,1,0,
		1,1,1,0,1,1,1,
		0,1,1,1,1,1,0
	};
	Mat census(3, 7, CV_8U, data);
	censusTransform32S(src, dest, census);
}

static void censusTrans32S_test(Mat & src, Mat & dest)
{
	/*uchar data[49] =
	{
		0,0,0,1,0,0,0,//1
		0,0,1,1,1,0,0,//3
		0,1,1,1,1,1,0,//5
		1,1,1,0,1,1,1,//6
		0,1,1,1,1,1,0,//5
		0,0,1,1,1,0,0,//3
		0,0,0,1,0,0,0,//1
	};//24
	Mat census(7, 7, CV_8U, data);*/
	/*uchar data[81] =
	{
		0,0,0,0,1,0,0,0,0,//1
		0,1,0,0,1,0,0,1,0,//1
		0,0,0,1,1,1,0,0,0,//5
		0,0,1,1,1,1,1,0,0,//5
		1,1,1,1,0,1,1,1,1,//8
		0,0,1,1,1,1,1,0,0,//5
		0,0,0,1,1,1,0,0,0,//5
		0,1,0,0,1,0,0,1,0,//1
		0,0,0,0,1,0,0,0,0,//1
	};//32
	Mat census(9, 9, CV_8U, data);*/

	/*uchar data[39] =
	{
		1,1,0,0,0,1,1,1,0,0,0,1,1,
		0,0,0,0,0,1,0,1,0,0,0,0,0,
		1,1,0,0,0,1,1,1,0,0,0,1,1
	};*/

	/*uchar data[27] =
	{
		1,1,0,1,1,1,0,1,1,
		0,0,0,1,0,1,0,0,0,
		1,1,0,1,1,1,0,1,1
	};*/
	uchar data[27] =
	{
		1,0,0,1,1,1,0,0,1,
		0,1,1,1,0,1,1,1,0,
		1,0,0,1,1,1,0,0,1
	};
	Mat census(3, 9, CV_8U, data);
	censusTransform32S(src, dest, census);
}

static void censusTrans8U_9x1(Mat & src, Mat & dest)
{
	if (dest.empty() || dest.depth() != CV_8U)dest.create(src.size(), CV_8U);
	const int w = src.cols;
	const int h = src.rows;

	const int r = 4;
	const int r2 = 2 * r;
	Mat im; copyMakeBorder(src, im, 0, 0, r, r, cv::BORDER_REPLICATE);

	uchar* s = im.ptr<uchar>(0); s += r;
	uchar* d = dest.ptr<uchar>(0);
	uchar val;//init value
	uchar* ss;
	for (int j = 0; j < h; j++)
	{
		for (int i = 0; i < w; i++)
		{
			val = 0;//init value
			ss = s - r;
			val = (*ss < *s) ? val | 1 : val; ss++; val <<= 1;
			val = (*ss < *s) ? val | 1 : val; ss++; val <<= 1;
			val = (*ss < *s) ? val | 1 : val; ss++; val <<= 1;
			val = (*ss < *s) ? val | 1 : val; ss++; val <<= 1;
			ss++;//skip r=0
			val = (*ss < *s) ? val | 1 : val; ss++; val <<= 1;
			val = (*ss < *s) ? val | 1 : val; ss++; val <<= 1;
			val = (*ss < *s) ? val | 1 : val; ss++; val <<= 1;
			val = (*ss < *s) ? val | 1 : val; ss++;

			*d = val;
			d++;
			s++;
		}
		s += r2;
	}
}

//0: gray image, 1: sobel/CENSUS image
void StereoMatch::computePrefilter(Mat & targetImage, Mat & referenceImage)
{
	const bool isColor = (pixelMatchingMethod % 2 == 1);
	if (isColor)
	{
		if (targetImage.channels() != 3 || referenceImage.channels() != 3)
		{
			std::cout << "input image must have 3 channels" << std::endl;
			CV_Assert(targetImage.channels() == 3);
			CV_Assert(referenceImage.channels() == 3);
		}

		target.resize(6);
		reference.resize(6);
		vector<Mat> temp;
		split(targetImage, temp);
		temp[0].copyTo(target[0]);
		temp[1].copyTo(target[2]);
		temp[2].copyTo(target[4]);
		split(referenceImage, temp);
		temp[0].copyTo(reference[0]);
		temp[1].copyTo(reference[2]);
		temp[2].copyTo(reference[4]);

		if (pixelMatchingMethod == CENSUS3x3Color)
		{
			censusTrans8U_3x3(target[0], target[1]);
			censusTrans8U_3x3(reference[0], reference[1]);
			censusTrans8U_3x3(target[2], target[3]);
			censusTrans8U_3x3(reference[2], reference[3]);
			censusTrans8U_3x3(target[4], target[5]);
			censusTrans8U_3x3(reference[4], reference[5]);
		}
		else if (pixelMatchingMethod == CENSUS9x1Color)
		{
			censusTrans8U_9x1(target[0], target[1]);
			censusTrans8U_9x1(reference[0], reference[1]);
			censusTrans8U_9x1(target[2], target[3]);
			censusTrans8U_9x1(reference[2], reference[3]);
			censusTrans8U_9x1(target[4], target[5]);
			censusTrans8U_9x1(reference[4], reference[5]);
		}
		else if (pixelMatchingMethod == CENSUS5x5Color)
		{
			censusTrans32S_5x5(target[0], target[1]);
			censusTrans32S_5x5(reference[0], reference[1]);
			censusTrans32S_5x5(target[2], target[3]);
			censusTrans32S_5x5(reference[2], reference[3]);
			censusTrans32S_5x5(target[4], target[5]);
			censusTrans32S_5x5(reference[4], reference[5]);
		}
		else if (pixelMatchingMethod == CENSUS7x5Color)
		{
			censusTrans32S_7x5_30(target[0], target[1]);
			censusTrans32S_7x5_30(reference[0], reference[1]);
			censusTrans32S_7x5_30(target[2], target[3]);
			censusTrans32S_7x5_30(reference[2], reference[3]);
			censusTrans32S_7x5_30(target[4], target[5]);
			censusTrans32S_7x5_30(reference[4], reference[5]);
		}
		else if (pixelMatchingMethod == CENSUS7x3Color)
		{
			censusTrans32S_7x3_16(target[0], target[1]);
			censusTrans32S_7x3_16(reference[0], reference[1]);
			censusTrans32S_7x3_16(target[2], target[3]);
			censusTrans32S_7x3_16(reference[2], reference[3]);
			censusTrans32S_7x3_16(target[4], target[5]);
			censusTrans32S_7x3_16(reference[4], reference[5]);
		}
		else if (pixelMatchingMethod == CENSUS13x3Color)
		{
			censusTrans32S_13x3_16(target[0], target[1]);
			censusTrans32S_13x3_16(reference[0], reference[1]);
			censusTrans32S_13x3_16(target[2], target[3]);
			censusTrans32S_13x3_16(reference[2], reference[3]);
			censusTrans32S_13x3_16(target[4], target[5]);
			censusTrans32S_13x3_16(reference[4], reference[5]);
		}
		else if (pixelMatchingMethod == CENSUSTESTColor)
		{
			censusTrans32S_test(target[0], target[1]);
			censusTrans32S_test(reference[0], reference[1]);
			censusTrans32S_test(target[2], target[3]);
			censusTrans32S_test(reference[2], reference[3]);
			censusTrans32S_test(target[4], target[5]);
			censusTrans32S_test(reference[4], reference[5]);
		}
		else
		{
			prefilterXSobel(target[0], target[1], preFilterCap);
			prefilterXSobel(reference[0], reference[1], preFilterCap);
			prefilterXSobel(target[2], target[3], preFilterCap);
			prefilterXSobel(reference[2], reference[3], preFilterCap);
			prefilterXSobel(target[4], target[5], preFilterCap);
			prefilterXSobel(reference[4], reference[5], preFilterCap);
		}
	}
	else
	{
		target.resize(2);
		reference.resize(2);

		if (targetImage.channels() == 3) cvtColor(targetImage, target[0], COLOR_BGR2GRAY);
		else targetImage.copyTo(target[0]);
		if (referenceImage.channels() == 3) cvtColor(referenceImage, reference[0], COLOR_BGR2GRAY);
		else referenceImage.copyTo(reference[0]);

		if (pixelMatchingMethod == CENSUS3x3)
		{
			censusTrans8U_3x3(target[0], target[1]);
			censusTrans8U_3x3(reference[0], reference[1]);
		}
		else if (pixelMatchingMethod == CENSUS9x1)
		{
			censusTrans8U_9x1(target[0], target[1]);
			censusTrans8U_9x1(reference[0], reference[1]);
		}
		else if (pixelMatchingMethod == CENSUS5x5)
		{
			censusTrans32S_5x5(target[0], target[1]);
			censusTrans32S_5x5(reference[0], reference[1]);
		}
		else if (pixelMatchingMethod == CENSUS7x5)
		{
			censusTrans32S_7x5_30(target[0], target[1]);
			censusTrans32S_7x5_30(reference[0], reference[1]);
		}
		else if (pixelMatchingMethod == CENSUS13x3)
		{
			censusTrans32S_13x3_16(target[0], target[1]);
			censusTrans32S_13x3_16(reference[0], reference[1]);
		}
		else if (pixelMatchingMethod == CENSUS7x3)
		{
			censusTrans32S_7x3_16(target[0], target[1]);
			censusTrans32S_7x3_16(reference[0], reference[1]);
		}
		else if (pixelMatchingMethod == CENSUSTEST)
		{
			censusTrans32S_test(target[0], target[1]);
			censusTrans32S_test(reference[0], reference[1]);
		}
		else
		{
			//prefilterGuided(target[0], target[1], preFilterCap);
			//prefilterGuided(reference[0], reference[1], preFilterCap);
			prefilterXSobel(target[0], target[1], preFilterCap);
			prefilterXSobel(reference[0], reference[1], preFilterCap);
		}
	}
}

void cvtColorBGR2GRAY_AVG(Mat & src, Mat & dest)
{
	CV_Assert(src.channels() == 3);
	dest.create(src.size(), CV_8U);
	const int size = src.size().area();
	uchar* s = src.ptr<uchar>();
	uchar* d = dest.ptr<uchar>();
	for (int i = 0; i < size; i++)
	{
		d[i] = saturate_cast<uchar>((s[3 * i + 0] + s[3 * i + 1] + s[3 * i + 2]) / 3.f);
	}
}

void StereoMatch::computeGuideImageForAggregation(Mat & input)
{
	if (input.channels() == 3) cvtColorBGR2GRAY_AVG(input, guideImage);
	else input.copyTo(guideImage);
	//cvtColor(input, guideImage, COLOR_BGR2BGRA);
	//Mat temp;
	//cv::decolor(input, guideImage, temp);
	//guidedImageFilter(guideImage, guideImage, guideImage, 2, 2);
	//guidedImageFilter(guideImage, guideImage, guideImage, 2, 2);
}
#pragma endregion

#pragma region cost computation of pixel matching

inline __m256i _mm256_squared_distance_epu8(__m256i src1, __m256i src2)
{
	__m256i s1 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(src1));
	__m256i s2 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(src2));
	__m256i sub = _mm256_sub_epi16(s1, s2);
	__m128i d1 = _mm256_cvtepi16_epu8(_mm256_mullo_epi16(sub, sub));
	s1 = _mm256_cvtepu8_epi16(_mm256_castsi256hi_si128(src1));
	s2 = _mm256_cvtepu8_epi16(_mm256_castsi256hi_si128(src2));
	sub = _mm256_sub_epi16(s1, s2);
	__m128i d2 = _mm256_cvtepi16_epu8(_mm256_mullo_epi16(sub, sub));
	return _mm256_set_m128i(d2, d1);
}

static void SDTruncate_8UC1(const Mat & src1, const Mat & src2, const int disparity, uchar thresh, Mat & dest)
{
	dest.create(src1.size(), CV_8U);

	const __m256i mtruncation = _mm256_set1_epi8(thresh);
	const int WIDTH = get_simd_floor(src1.cols, 32);
	if (disparity >= 0)
	{
		for (int j = 0; j < src1.rows; j++)
		{
			const uchar* s1 = src1.ptr<uchar>(j);
			const uchar* s2 = src2.ptr<uchar>(j);
			uchar* d = dest.ptr<uchar>(j);

			for (int i = 0; i < disparity; i++)
			{
				d[i] = std::min((s1[i] - s2[0]) * (s1[i] - s2[0]), (int)thresh);
			}
			for (int i = disparity; i < WIDTH; i += 32)
			{
				__m256i a = _mm256_loadu_si256((__m256i*)(s1 + i));
				__m256i b = _mm256_loadu_si256((__m256i*)(s2 - disparity + i));
				_mm256_storeu_si256((__m256i*)(d + i), _mm256_min_epu8(_mm256_squared_distance_epu8(a, b), mtruncation));
			}
			for (int i = WIDTH; i < src1.cols; i++)
			{
				d[i] = std::min((s1[i] - s2[i - disparity]) * (s1[i] - s2[i - disparity]), (int)thresh);
			}
		}
	}
}

static void SDTruncateBlend_8UC1(const Mat & src1, const Mat & src2, const Mat & src3, const Mat & src4, const int disparity, const uchar thresh, const float alpha, Mat & dest)
{
	dest.create(src1.size(), CV_8U);

	const __m256i mtruncation = _mm256_set1_epi8(thresh);
	const int WIDTH = get_simd_floor(src1.cols, 32);
	const int alpha_int = cvRound(255 * alpha);
	const __m256i ma = _mm256_set1_epi16(alpha_int << 7);

	if (disparity >= 0)
	{
		for (int j = 0; j < src1.rows; j++)
		{
			const uchar* s1 = src1.ptr<uchar>(j);
			const uchar* s2 = src2.ptr<uchar>(j);
			const uchar* s3 = src3.ptr<uchar>(j);
			const uchar* s4 = src4.ptr<uchar>(j);
			uchar* d = dest.ptr<uchar>(j);

			for (int i = 0; i < disparity; i++)
			{
				d[i] = saturate_cast<uchar>(alpha * std::min((s1[i] - s2[0]) * (s1[i] - s2[0]), (int)thresh)
					+ (1 - alpha) * std::min((s3[i] - s4[0]) * (s3[i] - s4[0]), (int)thresh));
			}
			for (int i = disparity; i < WIDTH; i += 32)
			{
				__m256i a = _mm256_loadu_si256((__m256i*)(s1 + i));
				__m256i b = _mm256_loadu_si256((__m256i*)(s2 - disparity + i));
				a = _mm256_min_epu8(_mm256_squared_distance_epu8(a, b), mtruncation);

				b = _mm256_loadu_si256((__m256i*)(s3 + i));
				__m256i c = _mm256_loadu_si256((__m256i*)(s4 - disparity + i));
				b = _mm256_min_epu8(_mm256_squared_distance_epu8(b, c), mtruncation);

				_mm256_storeu_si256((__m256i*)(d + i), _mm256_alphablend_epu8(a, b, ma));
			}
			for (int i = WIDTH; i < src1.cols; i++)
			{
				d[i] = saturate_cast<uchar>(alpha * std::min((s1[i] - s2[i - disparity]) * (s1[i] - s2[i - disparity]), (int)thresh)
					+ (1.f - alpha) * std::min((s3[i] - s4[i - disparity]) * (s3[i] - s4[i - disparity]), (int)thresh));
			}
		}
	}
}

static void ADTruncate_8UC1(const Mat & src1, const Mat & src2, const int disparity, const uchar thresh, Mat & dest)
{
	dest.create(src1.size(), CV_8U);

	const __m256i mtruncation = _mm256_set1_epi8(thresh);
	const int WIDTH = get_simd_floor(src1.cols, 32);
	if (disparity >= 0)
	{
		for (int j = 0; j < src1.rows; j++)
		{
			const uchar* s1 = src1.ptr<uchar>(j);
			const uchar* s2 = src2.ptr<uchar>(j);
			uchar* d = dest.ptr<uchar>(j);

			for (int i = 0; i < disparity; i++)
			{
				d[i] = std::min((uchar)abs(s1[i] - s2[0]), thresh);
			}
			for (int i = disparity; i < WIDTH; i += 32)
			{
				__m256i a = _mm256_loadu_si256((__m256i*)(s1 + i));
				__m256i b = _mm256_loadu_si256((__m256i*)(s2 - disparity + i));

				_mm256_storeu_si256((__m256i*)(d + i), _mm256_min_epu8(_mm256_adds_epu8(_mm256_subs_epu8(a, b), _mm256_subs_epu8(b, a)), mtruncation));
			}
			for (int i = WIDTH; i < src1.cols; i++)
			{
				d[i] = std::min((uchar)abs(s1[i] - s2[i - disparity]), thresh);
			}
		}
	}
}

static void ADTruncateBlend_8UC1(const Mat & src1, const Mat & src2, const Mat & src3, const Mat & src4, const int disparity, const uchar thresh, const float alpha, Mat & dest)
{
	dest.create(src1.size(), CV_8U);

	const __m256i mtruncation = _mm256_set1_epi8(thresh);
	const int WIDTH = get_simd_floor(src1.cols, 32);
	const int alpha_int = cvRound(255 * alpha);
	const __m256i ma = _mm256_set1_epi16(alpha_int << 7);

	if (disparity >= 0)
	{
		for (int j = 0; j < src1.rows; j++)
		{
			const uchar* s1 = src1.ptr<uchar>(j);
			const uchar* s2 = src2.ptr<uchar>(j);
			const uchar* s3 = src3.ptr<uchar>(j);
			const uchar* s4 = src4.ptr<uchar>(j);
			uchar* d = dest.ptr<uchar>(j);

			for (int i = 0; i < disparity; i++)
			{
				d[i] = saturate_cast<uchar>(alpha * std::min((uchar)abs(s1[i] - s2[0]), thresh)
					+ (1 - alpha) * std::min((uchar)abs(s3[i] - s4[0]), thresh));
			}
			for (int i = disparity; i < WIDTH; i += 32)
			{
				__m256i a = _mm256_loadu_si256((__m256i*)(s1 + i));
				__m256i b = _mm256_loadu_si256((__m256i*)(s2 - disparity + i));
				a = _mm256_min_epu8(_mm256_adds_epu8(_mm256_subs_epu8(a, b), _mm256_subs_epu8(b, a)), mtruncation);

				b = _mm256_loadu_si256((__m256i*)(s3 + i));
				__m256i c = _mm256_loadu_si256((__m256i*)(s4 - disparity + i));
				b = _mm256_min_epu8(_mm256_adds_epu8(_mm256_subs_epu8(b, c), _mm256_subs_epu8(c, b)), mtruncation);

				_mm256_storeu_si256((__m256i*)(d + i), _mm256_alphablend_epu8(a, b, ma));
			}
			for (int i = WIDTH; i < src1.cols; i++)
			{
				d[i] = saturate_cast<uchar>(alpha * std::min((uchar)abs(s1[i] - s2[i - disparity]), thresh)
					+ (1.f - alpha) * std::min((uchar)abs(s3[i] - s4[i - disparity]), thresh));
			}
		}
	}
}

static void BTTruncate_8UC1(const Mat & src1, const Mat & src2, const int disparity, const uchar thresh, Mat & dest)
{
	dest.create(src1.size(), CV_8U);
	const __m256i mtruncation = _mm256_set1_epi8(thresh);
	const int WIDTH = get_simd_floor(src1.cols, 32);
	const int DISPARITY = get_simd_floor(disparity, 32);

	if (disparity >= 0)
	{
		for (int j = 0; j < src1.rows; j++)
		{
			const uchar* s1 = src1.ptr<uchar>(j);
			const uchar* s2 = src2.ptr<uchar>(j);
			uchar* d = dest.ptr<uchar>(j);

			for (int i = 0; i < DISPARITY; i += 32)
			{
				__m256i v1 = _mm256_loadu_si256((__m256i*)(s1 + i));
				__m256i v2 = _mm256_loadu_si256((__m256i*)(s2 + 0));
				__m256i p2 = _mm256_avg_epu8(v2, _mm256_loadu_si256((__m256i*)(s2 + 1)));
				__m256i v2x = _mm256_max_epu8(p2, v2);
				__m256i v2n = _mm256_min_epu8(p2, v2);
				_mm256_storeu_si256((__m256i*)(d + i), _mm256_min_epu8(_mm256_max_epu8(_mm256_subs_epu8(v1, v2x), _mm256_subs_epu8(v2n, v1)), mtruncation));
			}
			for (int i = DISPARITY; i < disparity; i++)
			{
				int v1 = s1[i];
				int v2 = s2[0];
				int p2 = (s2[1] + v2) >> 1;
				int v2x = max(v2, p2);
				int v2n = min(v2, p2);
				int a = max(0, max(v1 - v2x, v2n - v1));
				d[i] = min(a, (int)thresh);
			}
			for (int i = disparity; i < WIDTH; i += 32)
			{
				__m256i v1 = _mm256_loadu_si256((__m256i*)(s1 + i));
				__m256i v2 = _mm256_loadu_si256((__m256i*)(s2 + i - disparity + 0));
				__m256i p2 = _mm256_avg_epu8(v2, _mm256_loadu_si256((__m256i*)(s2 + i - disparity + 1)));
				__m256i m2 = _mm256_avg_epu8(v2, _mm256_loadu_si256((__m256i*)(s2 + i - disparity - 1)));
				__m256i v2x = _mm256_max_epu8(_mm256_max_epu8(m2, p2), v2);
				__m256i v2n = _mm256_min_epu8(_mm256_min_epu8(m2, p2), v2);
				_mm256_storeu_si256((__m256i*)(d + i), _mm256_min_epu8(_mm256_max_epu8(_mm256_subs_epu8(v1, v2x), _mm256_subs_epu8(v2n, v1)), mtruncation));
			}
			for (int i = WIDTH; i < src1.cols; i++)
			{
				uchar v1 = s1[i];
				uchar v2 = (s2[i - disparity + 0]);
				uchar p2 = (s2[i - disparity + 1] + v2) >> 1;
				uchar m2 = (s2[i - disparity - 1] + v2) >> 1;
				uchar v2x = max(max(m2, p2), v2);
				uchar v2n = min(min(m2, p2), v2);

				uchar a = max(0, max(v1 - v2x, v2n - v1));
				d[i] = min(a, thresh);
			}
		}
	}
}

static void BTTruncateBlend_8UC1(const Mat & src1, const Mat & src2, const Mat & src3, const Mat & src4, const int disparity, uchar thresh, const float alpha, Mat & dest)
{
	dest.create(src1.size(), CV_8U);
	const __m256i mtruncation = _mm256_set1_epi8(thresh);
	const int WIDTH = get_simd_floor(src1.cols, 32);
	const int DISPARITY = get_simd_floor(disparity, 32);
	const int alpha_int = cvRound(255 * alpha);
	const __m256i ma = _mm256_set1_epi16(alpha_int << 7);

	if (disparity >= 0)
	{
		for (int j = 0; j < src1.rows; j++)
		{
			const uchar* s1 = src1.ptr<uchar>(j);
			const uchar* s2 = src2.ptr<uchar>(j);
			const uchar* s3 = src3.ptr<uchar>(j);
			const uchar* s4 = src4.ptr<uchar>(j);
			uchar* d = dest.ptr<uchar>(j);

			for (int i = 0; i < DISPARITY; i += 32)
			{
				__m256i v1 = _mm256_loadu_si256((__m256i*)(s1 + i));
				__m256i v2 = _mm256_loadu_si256((__m256i*)(s2));
				__m256i p2 = _mm256_avg_epu8(v2, _mm256_loadu_si256((__m256i*)(s2 + 1)));
				__m256i v2x = _mm256_max_epu8(p2, v2);
				__m256i v2n = _mm256_min_epu8(p2, v2);

				__m256i d1 = _mm256_min_epu8(_mm256_max_epu8(_mm256_subs_epu8(v1, v2x), _mm256_subs_epu8(v2n, v1)), mtruncation);

				v1 = _mm256_loadu_si256((__m256i*)(s3 + i));
				v2 = _mm256_loadu_si256((__m256i*)(s4));
				p2 = _mm256_avg_epu8(v2, _mm256_loadu_si256((__m256i*)(s4 + 1)));
				v2x = _mm256_max_epu8(p2, v2);
				v2n = _mm256_min_epu8(p2, v2);

				__m256i d2 = _mm256_min_epu8(_mm256_max_epu8(_mm256_subs_epu8(v1, v2x), _mm256_subs_epu8(v2n, v1)), mtruncation);

				_mm256_storeu_si256((__m256i*)(d + i), _mm256_alphablend_epu8(d1, d2, ma));
			}
			//for (int i = 0; i < disparity; i++)
			for (int i = DISPARITY; i < disparity; i++)
			{
				uchar v1 = s1[i];
				uchar v2 = (s2[0]);
				uchar p2 = (s2[1] + v2) >> 1;
				uchar v2x = max(p2, v2);
				uchar v2n = min(p2, v2);

				uchar d1 = std::min((uchar)std::max(0, std::max(v1 - v2x, v2n - v1)), thresh);

				v1 = s3[i];
				v2 = (s4[0]);
				p2 = (s4[1] + v2) >> 1;
				v2x = max(p2, v2);
				v2n = min(p2, v2);
				uchar d2 = std::min((uchar)std::max(0, std::max(v1 - v2x, v2n - v1)), thresh);
				d[i] = saturate_cast<uchar>(alpha * d1 + (1.f - alpha) * d2);
			}
			for (int i = disparity; i < WIDTH; i += 32)
			{
				__m256i v1 = _mm256_loadu_si256((__m256i*)(s1 + i));
				__m256i v2 = _mm256_loadu_si256((__m256i*)(s2 + i - disparity + 0));
				__m256i p2 = _mm256_avg_epu8(v2, _mm256_loadu_si256((__m256i*)(s2 + i - disparity + 1)));
				__m256i m2 = _mm256_avg_epu8(v2, _mm256_loadu_si256((__m256i*)(s2 + i - disparity - 1)));
				__m256i v2x = _mm256_max_epu8(_mm256_max_epu8(m2, p2), v2);
				__m256i v2n = _mm256_min_epu8(_mm256_min_epu8(m2, p2), v2);

				__m256i d1 = _mm256_min_epu8(_mm256_max_epu8(_mm256_subs_epu8(v1, v2x), _mm256_subs_epu8(v2n, v1)), mtruncation);

				v1 = _mm256_loadu_si256((__m256i*)(s3 + i));
				v2 = _mm256_loadu_si256((__m256i*)(s4 + i - disparity + 0));
				p2 = _mm256_avg_epu8(v2, _mm256_loadu_si256((__m256i*)(s4 + i - disparity + 1)));
				m2 = _mm256_avg_epu8(v2, _mm256_loadu_si256((__m256i*)(s4 + i - disparity - 1)));
				v2x = _mm256_max_epu8(_mm256_max_epu8(m2, p2), v2);
				v2n = _mm256_min_epu8(_mm256_min_epu8(m2, p2), v2);

				__m256i d2 = _mm256_min_epu8(_mm256_max_epu8(_mm256_subs_epu8(v1, v2x), _mm256_subs_epu8(v2n, v1)), mtruncation);

				_mm256_storeu_si256((__m256i*)(d + i), _mm256_alphablend_epu8(d1, d2, ma));
			}
			//for (int i = disparity; i < src1.cols; i++)
			for (int i = WIDTH; i < src1.cols; i++)
			{
				uchar v1 = s1[i];
				uchar v2 = (s2[i - disparity + 0]);
				uchar p2 = (s2[i - disparity + 1] + v2) >> 1;
				uchar m2 = (s2[i - disparity - 1] + v2) >> 1;
				uchar v2x = max(max(m2, p2), v2);
				uchar v2n = min(min(m2, p2), v2);

				uchar d1 = std::min((uchar)std::max(0, std::max(v1 - v2x, v2n - v1)), thresh);

				v1 = s3[i];
				v2 = (s4[i - disparity + 0]);
				p2 = (s4[i - disparity + 1] + v2) >> 1;
				m2 = (s4[i - disparity - 1] + v2) >> 1;
				v2x = max(max(m2, p2), v2);
				v2n = min(min(m2, p2), v2);
				uchar d2 = std::min((uchar)std::max(0, std::max(v1 - v2x, v2n - v1)), thresh);
				d[i] = saturate_cast<uchar>(alpha * d1 + (1.f - alpha) * d2);
			}
		}
	}
}

static void BTFullTruncate_8UC1(const Mat & src1, const Mat & src2, const int disparity, const uchar thresh, Mat & dest)
{
	dest.create(src1.size(), CV_8U);

	const __m256i mtruncation = _mm256_set1_epi8(thresh);
	const int WIDTH = get_simd_floor(src1.cols, 32);
	const int DISPARITY = get_simd_floor(disparity, 32);

	if (disparity >= 0)
	{
		for (int j = 0; j < src1.rows; j++)
		{
			const uchar* s1 = src1.ptr<uchar>(j);
			const uchar* s2 = src2.ptr<uchar>(j);
			uchar* d = dest.ptr<uchar>(j);

			for (int i = 0; i < DISPARITY; i += 32)
			{
				__m256i v1 = _mm256_loadu_si256((__m256i*)(s1 + i));
				__m256i p1 = _mm256_avg_epu8(v1, _mm256_loadu_si256((__m256i*)(s1 + i + 1)));
				__m256i m1 = _mm256_avg_epu8(v1, _mm256_loadu_si256((__m256i*)(s1 + max(i - 1, 0))));
				__m256i v1x = _mm256_max_epu8(_mm256_max_epu8(m1, p1), v1);
				__m256i v1n = _mm256_min_epu8(_mm256_min_epu8(m1, p1), v1);

				__m256i v2 = _mm256_loadu_si256((__m256i*)(s2 + 0));
				__m256i p2 = _mm256_avg_epu8(v2, _mm256_loadu_si256((__m256i*)(s2 + 1)));
				__m256i v2x = _mm256_max_epu8(p2, v2);
				__m256i v2n = _mm256_min_epu8(p2, v2);

				__m256i a = _mm256_max_epu8(_mm256_subs_epu8(v1, v2x), _mm256_subs_epu8(v2n, v1));
				__m256i b = _mm256_max_epu8(_mm256_subs_epu8(v2, v1x), _mm256_subs_epu8(v1n, v2));

				_mm256_storeu_si256((__m256i*)(d + i), _mm256_min_epu8(_mm256_min_epu8(a, b), mtruncation));
			}
			//for (int i = 0; i < disparity; i++)
			for (int i = DISPARITY; i < disparity; i++)
			{
				uchar v1 = s1[i];
				uchar p1 = (s1[i + 1] + v1) >> 1;
				uchar m1 = (s1[max(i - 1, 0)] + v1) >> 1;
				uchar v1x = max(max(m1, p1), v1);
				uchar v1n = min(min(m1, p1), v1);

				uchar v2 = s2[0];
				uchar p2 = (s2[1] + v2) >> 1;
				uchar v2x = max(p2, v2);
				uchar v2n = min(p2, v2);

				uchar a = max(0, max(v1 - v2x, v2n - v1));
				uchar b = max(0, max(v2 - v1x, v1n - v2));

				d[i] = min(min(a, b), thresh);
			}
			for (int i = disparity; i < WIDTH; i += 32)
			{
				__m256i v1 = _mm256_loadu_si256((__m256i*)(s1 + i));
				__m256i p1 = _mm256_avg_epu8(v1, _mm256_loadu_si256((__m256i*)(s1 + i + 1)));
				__m256i m1 = _mm256_avg_epu8(v1, _mm256_loadu_si256((__m256i*)(s1 + i - 1)));
				__m256i v1x = _mm256_max_epu8(_mm256_max_epu8(m1, p1), v1);
				__m256i v1n = _mm256_min_epu8(_mm256_min_epu8(m1, p1), v1);

				__m256i v2 = _mm256_loadu_si256((__m256i*)(s2 + i - disparity + 0));
				__m256i p2 = _mm256_avg_epu8(v2, _mm256_loadu_si256((__m256i*)(s2 + i - disparity + 1)));
				__m256i m2 = _mm256_avg_epu8(v2, _mm256_loadu_si256((__m256i*)(s2 + i - disparity - 1)));
				__m256i v2x = _mm256_max_epu8(_mm256_max_epu8(m2, p2), v2);
				__m256i v2n = _mm256_min_epu8(_mm256_min_epu8(m2, p2), v2);

				__m256i a = _mm256_max_epu8(_mm256_subs_epu8(v1, v2x), _mm256_subs_epu8(v2n, v1));
				__m256i b = _mm256_max_epu8(_mm256_subs_epu8(v2, v1x), _mm256_subs_epu8(v1n, v2));

				_mm256_storeu_si256((__m256i*)(d + i), _mm256_min_epu8(_mm256_min_epu8(a, b), mtruncation));
			}
			//for (int i = disparity; i < src1.cols; i++)
			for (int i = WIDTH; i < src1.cols; i++)
			{
				uchar v1 = s1[i];
				uchar p1 = (s1[i + 1] + v1) >> 1;
				uchar m1 = (s1[i - 1] + v1) >> 1;
				uchar v1x = max(max(m1, p1), v1);
				uchar v1n = min(min(m1, p1), v1);

				uchar v2 = (s2[i - disparity + 0]);
				uchar p2 = (s2[i - disparity + 1] + v2) >> 1;
				uchar m2 = (s2[i - disparity - 1] + v2) >> 1;
				uchar v2x = max(max(m2, p2), v2);
				uchar v2n = min(min(m2, p2), v2);

				uchar a = max(0, max(v1 - v2x, v2n - v1));
				uchar b = max(0, max(v2 - v1x, v1n - v2));

				d[i] = min(min(a, b), thresh);
			}
		}
	}
}

static void BTFullTruncateBlend_8UC1(const Mat & src1, const Mat & src2, const Mat & src3, const Mat & src4, const int disparity, uchar thresh, const float alpha, Mat & dest)
{
	dest.create(src1.size(), CV_8U);
	const __m256i mtruncation = _mm256_set1_epi8(thresh);
	const int WIDTH = get_simd_floor(src1.cols, 32);
	const int DISPARITY = get_simd_floor(disparity, 32);
	const int alpha_int = cvRound(255 * alpha);
	const __m256i ma = _mm256_set1_epi16(alpha_int << 7);

	if (disparity >= 0)
	{
		for (int j = 0; j < src1.rows; j++)
		{
			const uchar* s1 = src1.ptr<uchar>(j);
			const uchar* s2 = src2.ptr<uchar>(j);
			const uchar* s3 = src3.ptr<uchar>(j);
			const uchar* s4 = src4.ptr<uchar>(j);
			uchar* d = dest.ptr<uchar>(j);

			for (int i = 0; i < DISPARITY; i += 32)
			{
				__m256i v1 = _mm256_loadu_si256((__m256i*)(s1 + i + 0));
				__m256i p1 = _mm256_avg_epu8(v1, _mm256_loadu_si256((__m256i*)(s1 + i + 1)));
				__m256i m1 = _mm256_avg_epu8(v1, _mm256_loadu_si256((__m256i*)(s1 + max(i - 1, 0))));
				__m256i v1x = _mm256_max_epu8(_mm256_max_epu8(m1, p1), v1);
				__m256i v1n = _mm256_min_epu8(_mm256_min_epu8(m1, p1), v1);

				__m256i v2 = _mm256_loadu_si256((__m256i*)(s2 + 0));
				__m256i p2 = _mm256_avg_epu8(v2, _mm256_loadu_si256((__m256i*)(s2 + 1)));
				__m256i v2x = _mm256_max_epu8(p2, v2);
				__m256i v2n = _mm256_min_epu8(p2, v2);

				__m256i a = _mm256_max_epu8(_mm256_subs_epu8(v1, v2x), _mm256_subs_epu8(v2n, v1));
				__m256i b = _mm256_max_epu8(_mm256_subs_epu8(v2, v1x), _mm256_subs_epu8(v1n, v2));
				__m256i d1 = _mm256_min_epu8(_mm256_min_epu8(a, b), mtruncation);

				v1 = _mm256_loadu_si256((__m256i*)(s3 + i));
				p1 = _mm256_avg_epu8(v1, _mm256_loadu_si256((__m256i*)(s3 + i + 1)));
				m1 = _mm256_avg_epu8(v1, _mm256_loadu_si256((__m256i*)(s3 + i - 1)));
				v1x = _mm256_max_epu8(_mm256_max_epu8(m1, p1), v1);
				v1n = _mm256_min_epu8(_mm256_min_epu8(m1, p1), v1);

				v2 = _mm256_loadu_si256((__m256i*)(s4));
				p2 = _mm256_avg_epu8(v2, _mm256_loadu_si256((__m256i*)(s4 + 1)));
				v2x = _mm256_max_epu8(p2, v2);
				v2n = _mm256_min_epu8(p2, v2);

				a = _mm256_max_epu8(_mm256_subs_epu8(v1, v2x), _mm256_subs_epu8(v2n, v1));
				b = _mm256_max_epu8(_mm256_subs_epu8(v2, v1x), _mm256_subs_epu8(v1n, v2));
				__m256i d2 = _mm256_min_epu8(_mm256_min_epu8(a, b), mtruncation);

				_mm256_storeu_si256((__m256i*)(d + i), _mm256_alphablend_epu8(d1, d2, ma));
			}
			//for (int i = 0; i < disparity; i++)
			for (int i = DISPARITY; i < disparity; i++)
			{

				uchar v1 = (s1[i + 0]);
				uchar p1 = (s1[i + 1] + v1) >> 1;
				uchar m1 = (s1[max(i - 1, 0)] + v1) >> 1;
				uchar v1x = max(max(m1, p1), v1);
				uchar v1n = min(min(m1, p1), v1);

				uchar v2 = (s2[0]);
				uchar p2 = (s2[1] + v2) >> 1;
				uchar v2x = max(p2, v2);
				uchar v2n = min(p2, v2);

				uchar a = max(0, max(v1 - v2x, v2n - v1));
				uchar b = max(0, max(v2 - v1x, v1n - v2));
				uchar d1 = std::min((uchar)std::min(a, b), thresh);

				v1 = s3[i];
				p1 = (s3[i + 1] + v1) >> 1;
				m1 = (s3[i - 1] + v1) >> 1;
				v1x = max(max(m1, p1), v1);
				v1n = min(min(m1, p1), v1);

				v2 = (s4[0]);
				p2 = (s4[1] + v2) >> 1;
				v2x = max(p2, v2);
				v2n = min(p2, v2);

				a = max(0, max(v1 - v2x, v2n - v1));
				b = max(0, max(v2 - v1x, v1n - v2));
				uchar d2 = std::min((uchar)std::min(a, b), thresh);

				d[i] = saturate_cast<uchar>(alpha * d1 + (1.f - alpha) * d2);
			}
			for (int i = disparity; i < WIDTH; i += 32)
			{
				__m256i v1 = _mm256_loadu_si256((__m256i*)(s1 + i + 0));
				__m256i p1 = _mm256_avg_epu8(v1, _mm256_loadu_si256((__m256i*)(s1 + i + 1)));
				__m256i m1 = _mm256_avg_epu8(v1, _mm256_loadu_si256((__m256i*)(s1 + i - 1)));
				__m256i v1x = _mm256_max_epu8(_mm256_max_epu8(m1, p1), v1);
				__m256i v1n = _mm256_min_epu8(_mm256_min_epu8(m1, p1), v1);

				__m256i v2 = _mm256_loadu_si256((__m256i*)(s2 + i - disparity + 0));
				__m256i p2 = _mm256_avg_epu8(v2, _mm256_loadu_si256((__m256i*)(s2 + i - disparity + 1)));
				__m256i m2 = _mm256_avg_epu8(v2, _mm256_loadu_si256((__m256i*)(s2 + i - disparity - 1)));
				__m256i v2x = _mm256_max_epu8(_mm256_max_epu8(m2, p2), v2);
				__m256i v2n = _mm256_min_epu8(_mm256_min_epu8(m2, p2), v2);

				__m256i a = _mm256_max_epu8(_mm256_subs_epu8(v1, v2x), _mm256_subs_epu8(v2n, v1));
				__m256i b = _mm256_max_epu8(_mm256_subs_epu8(v2, v1x), _mm256_subs_epu8(v1n, v2));
				__m256i d1 = _mm256_min_epu8(_mm256_min_epu8(a, b), mtruncation);

				v1 = _mm256_loadu_si256((__m256i*)(s3 + i));
				p1 = _mm256_avg_epu8(v1, _mm256_loadu_si256((__m256i*)(s3 + i + 1)));
				m1 = _mm256_avg_epu8(v1, _mm256_loadu_si256((__m256i*)(s3 + i - 1)));
				v1x = _mm256_max_epu8(_mm256_max_epu8(m1, p1), v1);
				v1n = _mm256_min_epu8(_mm256_min_epu8(m1, p1), v1);

				v2 = _mm256_loadu_si256((__m256i*)(s4 + i - disparity + 0));
				p2 = _mm256_avg_epu8(v2, _mm256_loadu_si256((__m256i*)(s4 + i - disparity + 1)));
				m2 = _mm256_avg_epu8(v2, _mm256_loadu_si256((__m256i*)(s4 + i - disparity - 1)));
				v2x = _mm256_max_epu8(_mm256_max_epu8(m2, p2), v2);
				v2n = _mm256_min_epu8(_mm256_min_epu8(m2, p2), v2);

				a = _mm256_max_epu8(_mm256_subs_epu8(v1, v2x), _mm256_subs_epu8(v2n, v1));
				b = _mm256_max_epu8(_mm256_subs_epu8(v2, v1x), _mm256_subs_epu8(v1n, v2));
				__m256i d2 = _mm256_min_epu8(_mm256_min_epu8(a, b), mtruncation);

				_mm256_storeu_si256((__m256i*)(d + i), _mm256_alphablend_epu8(d1, d2, ma));
			}
			//for (int i = disparity; i < src1.cols; i++)
			for (int i = WIDTH; i < src1.cols; i++)
			{
				uchar v1 = (s1[i + 0]);
				uchar p1 = (s1[i + 1] + v1) >> 1;
				uchar m1 = (s1[i - 1] + v1) >> 1;
				uchar v1x = max(max(m1, p1), v1);
				uchar v1n = min(min(m1, p1), v1);

				uchar v2 = (s2[i - disparity + 0]);
				uchar p2 = (s2[i - disparity + 1] + v2) >> 1;
				uchar m2 = (s2[i - disparity - 1] + v2) >> 1;
				uchar v2x = max(max(m2, p2), v2);
				uchar v2n = min(min(m2, p2), v2);

				uchar a = max(0, max(v1 - v2x, v2n - v1));
				uchar b = max(0, max(v2 - v1x, v1n - v2));
				uchar d1 = std::min((uchar)std::min(a, b), thresh);

				v1 = s3[i];
				p1 = (s3[i + 1] + v1) >> 1;
				m1 = (s3[i - 1] + v1) >> 1;
				v1x = max(max(m1, p1), v1);
				v1n = min(min(m1, p1), v1);
				v2 = (s4[i - disparity + 0]);
				p2 = (s4[i - disparity + 1] + v2) >> 1;
				m2 = (s4[i - disparity - 1] + v2) >> 1;
				v2x = max(max(m2, p2), v2);
				v2n = min(min(m2, p2), v2);

				a = max(0, max(v1 - v2x, v2n - v1));
				b = max(0, max(v2 - v1x, v1n - v2));
				uchar d2 = std::min((uchar)std::min(a, b), thresh);

				d[i] = saturate_cast<uchar>(alpha * d1 + (1.f - alpha) * d2);
			}
		}
	}
}

//simple stereo matching implementaion
template<typename srcType>
static void HammingDistance32S_8UC1(Mat & src1, Mat & src2, const int disparity, Mat & dest)
{
	if (dest.empty())dest.create(src1.size(), CV_8U);
	const int WIDTH = get_simd_floor(src1.cols, 32);
	const int h = src1.rows;

	for (int j = 0; j < h; j++)
	{
		srcType* s1 = src1.ptr<srcType>(j);
		srcType* s2 = src2.ptr<srcType>(j);
		uchar* d = dest.ptr<uchar>(j);
		for (int i = 0; i < disparity; i++)
		{
			d[i] = _mm_popcnt_u32((s1[i] ^ s2[0]));
		}
		/* for AVX512
					for (int i = disparity; i < WIDTH; i += 32)
					{
						__m256i ms1 = _mm256_loadu_si256((__m256i*)(s1+i));
						__m256i ms2 = _mm256_loadu_si256((__m256i*)(s2+i-disparity));
						__m256i md = _mm256_xor_si256(ms1, ms2);
						_mm256_storeu_si256((__m256i*)(d+i), _mm256_popcnt_epi8(md));
						//_mm256_storeu_si256((__m256i*)(d + i), md);
					}
					for (int i = WIDTH; i < src1.cols; i++)
						*/
		for (int i = disparity; i < src1.cols; i++)
		{
			d[i] = _mm_popcnt_u32((s1[i] ^ s2[i - disparity]));
		}
	}

}

string StereoMatch::getCostMethodName(const Cost method)
{
	string mes = "";
	switch (method)
	{
	case SD:					mes = "SD"; break;
	case SDEdge:				mes = "SDEdge"; break;
	case SDEdgeBlend:			mes = "SDEdgeBlend"; break;
	case AD:					mes = "AD"; break;
	case ADEdge:				mes = "ADEdge"; break;
	case ADEdgeBlend:			mes = "ADEdgeBlend"; break;
	case BT:					mes = "BT"; break;
	case BTEdge:				mes = "BTEdge"; break;
	case BTEdgeBlend:			mes = "BTEdgeBlend"; break;
	case BTFull:				mes = "BTFull"; break;
	case BTFullEdge:			mes = "BTFullSobel"; break;
	case BTFullEdgeBlend:		mes = "BTFullSobelBlend"; break;
	case CENSUS3x3:				mes = "CENSUS3x3"; break;
	case CENSUS5x5:				mes = "CENSUS5x5"; break;
	case CENSUS7x5:				mes = "CENSUS7x5"; break;
	case CENSUS9x1:				mes = "CENSUS9x1"; break;
	case CENSUS13x3:			mes = "CENSUS13x3_16"; break;
	case CENSUS7x3:				mes = "CENSUS7x3_16"; break;
	case CENSUSTEST:			mes = "CENSUSTest"; break;

	case SDColor:				mes = "SDColor"; break;
	case SDEdgeColor:			mes = "SDEdgeColor"; break;
	case SDEdgeBlendColor:		mes = "SDEdgeBlendColor"; break;
	case ADColor:				mes = "ADColor"; break;
	case ADEdgeColor:			mes = "ADEdgeColor"; break;
	case ADEdgeBlendColor:		mes = "ADEdgeBlendColor"; break;
	case BTColor:				mes = "BTColor"; break;
	case BTEdgeColor:			mes = "BTEdgeColor"; break;
	case BTEdgeBlendColor:		mes = "BTEdgeBlendColor"; break;
	case BTFullColor:			mes = "BTFullColor"; break;
	case BTFullEdgeColor:		mes = "BTFullEdgeColor"; break;
	case BTFullEdgeBlendColor:	mes = "BTFullEdgeBlendColor"; break;
	case CENSUS3x3Color:		mes = "CENSUS3x3Color"; break;
	case CENSUS5x5Color:		mes = "CENSUS5x5Color"; break;
	case CENSUS7x5Color:		mes = "CENSUS7x5Color"; break;
	case CENSUS9x1Color:		mes = "CENSUS9x1Color"; break;
	case CENSUS13x3Color:		mes = "CENSUS13x3Color_16"; break;
	case CENSUS7x3Color:		mes = "CENSUS7x3Color_16"; break;
	case CENSUSTESTColor:		mes = "CENSUSTESTColor"; break;

		//case Pixel_Matching_SAD_TextureBlend:	mes = "Cost_Computation_SAD_TextureBlend"; break;
		//case Pixel_Matching_BT_TextureBlend:	mes = "Cost_Computation_BTTextureBlend"; break;
	default:					mes = "This cost computation method is not supported"; break;
	}
	return mes;
}

void  StereoMatch::setCostMethod(const Cost method)
{
	pixelMatchingMethod = method;
}

void StereoMatch::setCostColorDistance(const ColorDistance method)
{
	color_distance = method;
}

string StereoMatch::getCostColorDistanceName(ColorDistance method)
{
	std::string ret;
	switch (method)
	{
	case StereoMatch::ADD: ret = "ADD";
		break;
	case StereoMatch::AVG: ret = "AVG";
		break;
	case StereoMatch::MIN: ret = "MIN";
		break;
	case StereoMatch::MAX: ret = "MAX";
		break;
	default:
		ret = "not supported";
		break;
	}
	return ret;
}

void StereoMatch::computePixelMatchingCost(const int d, Mat & dest)
{
	//gray
	if (pixelMatchingMethod == SD)
	{
		SDTruncate_8UC1(target[0], reference[0], d, pixelMatchErrorCap, dest);
	}
	else if (pixelMatchingMethod == SDEdge)
	{
		SDTruncate_8UC1(target[1], reference[1], d, pixelMatchErrorCap, dest);
	}
	else if (pixelMatchingMethod == SDEdgeBlend)
	{
		SDTruncateBlend_8UC1(target[0], reference[0], target[1], reference[1], d, pixelMatchErrorCap, costAlphaImageSobel / 100.f, dest);
	}
	else if (pixelMatchingMethod == AD)
	{
		ADTruncate_8UC1(target[0], reference[0], d, pixelMatchErrorCap, dest);
	}
	else if (pixelMatchingMethod == ADEdge)
	{
		ADTruncate_8UC1(target[1], reference[1], d, pixelMatchErrorCap, dest);
	}
	else if (pixelMatchingMethod == ADEdgeBlend)
	{
		ADTruncateBlend_8UC1(target[0], reference[0], target[1], reference[1], d, pixelMatchErrorCap, costAlphaImageSobel / 100.f, dest);
	}
	else if (pixelMatchingMethod == BT)
	{
		BTTruncate_8UC1(target[0], reference[0], d, pixelMatchErrorCap, dest);
	}
	else if (pixelMatchingMethod == BTEdge)
	{
		BTTruncate_8UC1(target[1], reference[1], d, pixelMatchErrorCap, dest);
	}
	else if (pixelMatchingMethod == BTEdgeBlend)
	{
		BTTruncateBlend_8UC1(target[0], reference[0], target[1], reference[1], d, pixelMatchErrorCap, costAlphaImageSobel / 100.f, dest);
	}
	else if (pixelMatchingMethod == BTFull)
	{
		BTFullTruncate_8UC1(target[0], reference[0], d, pixelMatchErrorCap, dest);
	}
	else if (pixelMatchingMethod == BTFullEdge)
	{
		BTFullTruncate_8UC1(target[1], reference[1], d, pixelMatchErrorCap, dest);
	}
	else if (pixelMatchingMethod == BTFullEdgeBlend)
	{
		BTFullTruncateBlend_8UC1(target[0], reference[0], target[1], reference[1], d, pixelMatchErrorCap, costAlphaImageSobel / 100.f, dest);
	}
	else if (pixelMatchingMethod == CENSUS3x3 || pixelMatchingMethod == CENSUS9x1)
	{
		HammingDistance32S_8UC1<uchar>(target[1], reference[1], d, dest);
	}
	else if (pixelMatchingMethod == CENSUS5x5 || pixelMatchingMethod == CENSUS7x5 || pixelMatchingMethod == CENSUS13x3 || pixelMatchingMethod == CENSUS7x3 || pixelMatchingMethod == CENSUSTEST)
	{
		HammingDistance32S_8UC1<int>(target[1], reference[1], d, dest);
	}

	//color
	else if (pixelMatchingMethod == SDColor)
	{
		Mat temp;
		SDTruncate_8UC1(target[0], reference[0], d, pixelMatchErrorCap, dest);
		SDTruncate_8UC1(target[2], reference[2], d, pixelMatchErrorCap, temp);
		if (color_distance == ADD || color_distance == AVG) add(dest, temp, dest);
		else if (color_distance == MIN)min(dest, temp, dest);
		else if (color_distance == MAX)max(dest, temp, dest);
		SDTruncate_8UC1(target[4], reference[4], d, pixelMatchErrorCap, temp);
		if (color_distance == ADD || color_distance == AVG) add(dest, temp, dest);
		else if (color_distance == MIN)min(dest, temp, dest);
		else if (color_distance == MAX)max(dest, temp, dest);
		if (color_distance == AVG)divide(dest, 3, dest);

	}
	else if (pixelMatchingMethod == SDEdgeColor)
	{
		Mat temp;
		SDTruncate_8UC1(target[1], reference[1], d, pixelMatchErrorCap, dest);
		SDTruncate_8UC1(target[3], reference[3], d, pixelMatchErrorCap, temp);
		if (color_distance == ADD || color_distance == AVG) add(dest, temp, dest);
		else if (color_distance == MIN)min(dest, temp, dest);
		else if (color_distance == MAX)max(dest, temp, dest);
		SDTruncate_8UC1(target[5], reference[5], d, pixelMatchErrorCap, temp);
		if (color_distance == ADD || color_distance == AVG) add(dest, temp, dest);
		else if (color_distance == MIN)min(dest, temp, dest);
		else if (color_distance == MAX)max(dest, temp, dest);
		if (color_distance == AVG)divide(dest, 3, dest);
	}
	else if (pixelMatchingMethod == SDEdgeBlendColor)
	{
		Mat temp;
		SDTruncateBlend_8UC1(target[0], reference[0], target[1], reference[1], d, pixelMatchErrorCap, costAlphaImageSobel / 100.f, dest);
		SDTruncateBlend_8UC1(target[2], reference[2], target[3], reference[3], d, pixelMatchErrorCap, costAlphaImageSobel / 100.f, temp);
		if (color_distance == ADD || color_distance == AVG) add(dest, temp, dest);
		else if (color_distance == MIN)min(dest, temp, dest);
		else if (color_distance == MAX)max(dest, temp, dest);
		SDTruncateBlend_8UC1(target[4], reference[4], target[5], reference[5], d, pixelMatchErrorCap, costAlphaImageSobel / 100.f, temp);
		if (color_distance == ADD || color_distance == AVG) add(dest, temp, dest);
		else if (color_distance == MIN)min(dest, temp, dest);
		else if (color_distance == MAX)max(dest, temp, dest);
		if (color_distance == AVG)divide(dest, 3, dest);
	}
	else if (pixelMatchingMethod == ADColor)
	{
		Mat temp;
		ADTruncate_8UC1(target[0], reference[0], d, pixelMatchErrorCap, dest);
		ADTruncate_8UC1(target[2], reference[2], d, pixelMatchErrorCap, temp);
		if (color_distance == ADD || color_distance == AVG) add(dest, temp, dest);
		else if (color_distance == MIN)min(dest, temp, dest);
		else if (color_distance == MAX)max(dest, temp, dest);
		ADTruncate_8UC1(target[4], reference[4], d, pixelMatchErrorCap, temp);
		if (color_distance == ADD || color_distance == AVG) add(dest, temp, dest);
		else if (color_distance == MIN)min(dest, temp, dest);
		else if (color_distance == MAX)max(dest, temp, dest);
		if (color_distance == AVG)divide(dest, 3, dest);
	}
	else if (pixelMatchingMethod == ADEdgeColor)
	{
		Mat temp;
		ADTruncate_8UC1(target[1], reference[1], d, pixelMatchErrorCap, dest);
		ADTruncate_8UC1(target[3], reference[3], d, pixelMatchErrorCap, temp);
		if (color_distance == ADD || color_distance == AVG) add(dest, temp, dest);
		else if (color_distance == MIN)min(dest, temp, dest);
		else if (color_distance == MAX)max(dest, temp, dest);
		ADTruncate_8UC1(target[5], reference[5], d, pixelMatchErrorCap, temp);
		if (color_distance == ADD || color_distance == AVG) add(dest, temp, dest);
		else if (color_distance == MIN)min(dest, temp, dest);
		else if (color_distance == MAX)max(dest, temp, dest);
		if (color_distance == AVG)divide(dest, 3, dest);
	}
	else if (pixelMatchingMethod == ADEdgeBlendColor)
	{
		Mat temp;
		ADTruncateBlend_8UC1(target[0], reference[0], target[1], reference[1], d, pixelMatchErrorCap, costAlphaImageSobel / 100.f, dest);
		ADTruncateBlend_8UC1(target[2], reference[2], target[3], reference[3], d, pixelMatchErrorCap, costAlphaImageSobel / 100.f, temp);
		if (color_distance == ADD || color_distance == AVG) add(dest, temp, dest);
		else if (color_distance == MIN)min(dest, temp, dest);
		else if (color_distance == MAX)max(dest, temp, dest);
		ADTruncateBlend_8UC1(target[4], reference[4], target[5], reference[5], d, pixelMatchErrorCap, costAlphaImageSobel / 100.f, temp);
		if (color_distance == ADD || color_distance == AVG) add(dest, temp, dest);
		else if (color_distance == MIN)min(dest, temp, dest);
		else if (color_distance == MAX)max(dest, temp, dest);
		if (color_distance == AVG)divide(dest, 3, dest);
	}
	else if (pixelMatchingMethod == BTColor)
	{
		Mat temp;
		BTTruncate_8UC1(target[0], reference[0], d, pixelMatchErrorCap, dest);
		BTTruncate_8UC1(target[2], reference[2], d, pixelMatchErrorCap, temp);
		if (color_distance == ADD || color_distance == AVG) add(dest, temp, dest);
		else if (color_distance == MIN)min(dest, temp, dest);
		else if (color_distance == MAX)max(dest, temp, dest);
		BTTruncate_8UC1(target[4], reference[4], d, pixelMatchErrorCap, temp);
		if (color_distance == ADD || color_distance == AVG) add(dest, temp, dest);
		else if (color_distance == MIN)min(dest, temp, dest);
		else if (color_distance == MAX)max(dest, temp, dest);
		if (color_distance == AVG)divide(dest, 3, dest);
	}
	else if (pixelMatchingMethod == BTEdgeColor)
	{
		Mat temp;
		BTTruncate_8UC1(target[1], reference[1], d, pixelMatchErrorCap, dest);
		BTTruncate_8UC1(target[3], reference[3], d, pixelMatchErrorCap, temp);
		if (color_distance == ADD || color_distance == AVG) add(dest, temp, dest);
		else if (color_distance == MIN)min(dest, temp, dest);
		else if (color_distance == MAX)max(dest, temp, dest);
		BTTruncate_8UC1(target[5], reference[5], d, pixelMatchErrorCap, temp);
		add(dest, temp, dest);
	}
	else if (pixelMatchingMethod == BTEdgeBlendColor)
	{
		Mat temp;
		BTTruncateBlend_8UC1(target[0], reference[0], target[1], reference[1], d, pixelMatchErrorCap, costAlphaImageSobel / 100.f, dest);
		BTTruncateBlend_8UC1(target[2], reference[2], target[3], reference[3], d, pixelMatchErrorCap, costAlphaImageSobel / 100.f, temp);
		if (color_distance == ADD || color_distance == AVG) add(dest, temp, dest);
		else if (color_distance == MIN)min(dest, temp, dest);
		else if (color_distance == MAX)max(dest, temp, dest);
		BTTruncateBlend_8UC1(target[4], reference[4], target[5], reference[5], d, pixelMatchErrorCap, costAlphaImageSobel / 100.f, temp);
		if (color_distance == ADD || color_distance == AVG) add(dest, temp, dest);
		else if (color_distance == MIN)min(dest, temp, dest);
		else if (color_distance == MAX)max(dest, temp, dest);
		if (color_distance == AVG)divide(dest, 3, dest);
	}
	else if (pixelMatchingMethod == BTFullColor)
	{
		Mat temp;
		BTFullTruncate_8UC1(target[0], reference[0], d, pixelMatchErrorCap, dest);
		BTFullTruncate_8UC1(target[2], reference[2], d, pixelMatchErrorCap, temp);
		if (color_distance == ADD || color_distance == AVG) add(dest, temp, dest);
		else if (color_distance == MIN)min(dest, temp, dest);
		else if (color_distance == MAX)max(dest, temp, dest);
		BTFullTruncate_8UC1(target[4], reference[4], d, pixelMatchErrorCap, temp);
		if (color_distance == ADD || color_distance == AVG) add(dest, temp, dest);
		else if (color_distance == MIN)min(dest, temp, dest);
		else if (color_distance == MAX)max(dest, temp, dest);
		if (color_distance == AVG)divide(dest, 3, dest);
	}
	else if (pixelMatchingMethod == BTFullEdgeColor)
	{
		Mat temp;
		BTFullTruncate_8UC1(target[1], reference[1], d, pixelMatchErrorCap, dest);
		BTFullTruncate_8UC1(target[3], reference[3], d, pixelMatchErrorCap, temp);
		if (color_distance == ADD || color_distance == AVG) add(dest, temp, dest);
		else if (color_distance == MIN)min(dest, temp, dest);
		else if (color_distance == MAX)max(dest, temp, dest);
		BTFullTruncate_8UC1(target[5], reference[5], d, pixelMatchErrorCap, temp);
		if (color_distance == ADD || color_distance == AVG) add(dest, temp, dest);
		else if (color_distance == MIN)min(dest, temp, dest);
		else if (color_distance == MAX)max(dest, temp, dest);
		if (color_distance == AVG)divide(dest, 3, dest);
	}
	else if (pixelMatchingMethod == BTFullEdgeBlendColor)
	{
		Mat temp;
		BTFullTruncateBlend_8UC1(target[0], reference[0], target[1], reference[1], d, pixelMatchErrorCap, costAlphaImageSobel / 100.f, dest);
		BTFullTruncateBlend_8UC1(target[2], reference[2], target[3], reference[3], d, pixelMatchErrorCap, costAlphaImageSobel / 100.f, temp);
		if (color_distance == ADD || color_distance == AVG) add(dest, temp, dest);
		else if (color_distance == MIN)min(dest, temp, dest);
		else if (color_distance == MAX)max(dest, temp, dest);
		BTFullTruncateBlend_8UC1(target[4], reference[4], target[5], reference[5], d, pixelMatchErrorCap, costAlphaImageSobel / 100.f, temp);
		add(dest, temp, dest);
	}
	else if (pixelMatchingMethod == CENSUS3x3Color || pixelMatchingMethod == CENSUS9x1Color)
	{
		Mat temp;
		HammingDistance32S_8UC1<uchar>(target[1], reference[1], d, dest);
		HammingDistance32S_8UC1<uchar>(target[3], reference[3], d, temp);
		if (color_distance == ADD || color_distance == AVG) add(dest, temp, dest);
		else if (color_distance == MIN)min(dest, temp, dest);
		else if (color_distance == MAX)max(dest, temp, dest);
		HammingDistance32S_8UC1<uchar>(target[5], reference[5], d, temp);
		if (color_distance == ADD || color_distance == AVG) add(dest, temp, dest);
		else if (color_distance == MIN)min(dest, temp, dest);
		else if (color_distance == MAX)max(dest, temp, dest);
		if (color_distance == AVG)divide(dest, 3, dest);
	}
	else if (pixelMatchingMethod == CENSUS5x5Color || pixelMatchingMethod == CENSUS7x5Color || pixelMatchingMethod == CENSUS13x3Color || pixelMatchingMethod == CENSUS7x3Color || pixelMatchingMethod == CENSUSTESTColor)
	{
		Mat temp;
		HammingDistance32S_8UC1<int>(target[1], reference[1], d, dest);
		HammingDistance32S_8UC1<int>(target[3], reference[3], d, temp);
		if (color_distance == ADD || color_distance == AVG) add(dest, temp, dest);
		else if (color_distance == MIN)min(dest, temp, dest);
		else if (color_distance == MAX)max(dest, temp, dest);
		HammingDistance32S_8UC1<int>(target[5], reference[5], d, temp);
		if (color_distance == ADD || color_distance == AVG) add(dest, temp, dest);
		else if (color_distance == MIN)min(dest, temp, dest);
		else if (color_distance == MAX)max(dest, temp, dest);
		if (color_distance == AVG)divide(dest, 3, dest);
	}
	/*else if (PixelMatchingMethod == Pixel_Matching_SAD_TextureBlend)
	{
		Mat alpha;
		textureAlpha(target[0], alpha, sobelBlendMapParam2, sobelBlendMapParam1, sobelBlendMapParam_Size);
		getPixelMatchingCostSADAlpha(target, reference, alpha, d, dest);
	}
	else if (PixelMatchingMethod == Pixel_Matching_BT_TextureBlend)
	{
		Mat alpha;
		textureAlpha(target[0], alpha, sobelBlendMapParam2, sobelBlendMapParam1, sobelBlendMapParam_Size);
		getPixelMatchingCostBTAlpha(target, reference, alpha, d, dest);
	}*/
	else
	{
		cout << "This pixel matching method is not supported." << endl;
	}
}


inline float distance_functionEXP(float diff, float clip)
{
	return 1.f - exp(diff * diff / (-2.f * clip * clip));
}
inline float distance_functionL1(float diff, float clip)
{
	return min(abs(diff), clip);
}
inline float distance_functionL2(float diff, float clip)
{
	return min(diff * diff, clip * clip);
}

void StereoMatch::addCostIterativeFeedback(cv::Mat & cost, const int current_disparity, const cv::Mat & disparity, const int functionType, const int clip, float amp)
{
	CV_Assert(disparity.depth() == CV_16S);
	const short* dptr = disparity.ptr<short>();
	uchar* cptr = cost.ptr<uchar>();

	if (functionType == 2)
	{
		for (int i = 0; i < cost.size().area(); i++)
		{
			cptr[i] = saturate_cast<uchar>(cptr[i] + amp * distance_functionL2((float)current_disparity - dptr[i] / 16.f, (float)clip));
		}
	}
	else if (functionType == 1)
	{
		for (int i = 0; i < cost.size().area(); i++)
		{
			cptr[i] = saturate_cast<uchar>(cptr[i] + amp * distance_functionL1((float)current_disparity - dptr[i] / 16.f, (float)clip));
		}
	}
	else if (functionType == 0)
	{
		for (int i = 0; i < cost.size().area(); i++)
		{
			cptr[i] = saturate_cast<uchar>(cptr[i] + amp * distance_functionEXP((float)current_disparity - dptr[i] / 16.f, (float)clip));
		}
	}
	else
	{
		std::cout << "not supported for this function type in feedback" << std::endl;
	}
}

void StereoMatch::computetextureAlpha(Mat & src, Mat & dest, const int th1, const int th2, const int r)
{
	if (dest.empty())dest.create(src.size(), CV_8U);
	Mat temp;
	Sobel(src, temp, CV_16S, 1, 0);
	Mat temp2;

	convertScaleAbs(temp, temp2);
	//maxFilter(temp2,temp2,Size(2*r+1,2*r+1));
	//blur(temp2,temp2,Size(2*r+1,2*r+1));
	temp2 -= th1;

	max(temp2, th2, temp2);
	Mat temp33;
	temp2.convertTo(temp33, CV_32F, 1.0 / (double)th2);
	multiply(temp33, temp33, temp33, 1);
	temp33.convertTo(temp2, CV_8U, 255.0 / th2);
	//temp2*=255.0/(double)th2;
	//blur(temp2,temp2,Size(2*r+1,2*r+1));
	//threshold(temp2,dest,th,255,cv::THRESH_BINARY);

	maxFilter(temp2, dest, Size(aggregationRadiusH, 2 * r + 1));

	//blur(temp2,dest,size(2*r+1,2*r+1));
	//dest*=1.5;
	//GaussianBlur(dest,dest,Size(2*r+1,2*r+1),r/1.5);

	//imshow("texture", dest);
}

void StereoMatch::computePixelMatchingCostADAlpha(vector<Mat> &t, vector<Mat> &r, Mat & alpha, const int d, Mat & dest)
{
	;
}

void StereoMatch::computePixelMatchingCostBTAlpha(vector<Mat> &target, vector<Mat> &refference, Mat & alpha, const int d, Mat & dest)
{
	;
}
#pragma endregion

#pragma region cost aggregation

string StereoMatch::getAggregationMethodName(const Aggregation method)
{
	string mes;
	switch (method)
	{
	case Box:				mes = "Box"; break;
	case BoxShiftable:		mes = "BoxShiftable"; break;
	case Gaussian:			mes = "Gauss"; break;
	case GaussianShiftable:	mes = "GaussShiftable"; break;
	case Guided:			mes = "Guided"; break;
	case CrossBasedBox:		mes = "CrossBasedBox"; break;
	case Bilateral:			mes = "Bilateral"; break;

	default:				mes = "This cost computation method is not supported"; break;
	}
	return mes;
}

void StereoMatch::setAggregationMethod(const Aggregation method)
{
	aggregationMethod = method;
}

void StereoMatch::computeCostAggregation(Mat & src, Mat & dest, cv::InputArray guideImage)
{
	if (aggregationRadiusH == 0 && aggregationRadiusV == 0)
	{
		if (src.data != dest.data) src.copyTo(dest);
		return;
	}
	const float sigma_s_h = (float)aggregationSigmaSpace;
	const float sigma_s_v = (float)aggregationSigmaSpace;

	Mat guide = guideImage.getMat();

	//GaussianBlur(dsi, DSI[i], Size(SADWindowSize, SADWindowSize), 3);
	Size kernelSize = Size(2 * aggregationRadiusH + 1, 2 * aggregationRadiusV + 1);

	if (aggregationMethod == Box)
	{
		boxFilter(src, dest, -1, kernelSize);
	}
	else if (aggregationMethod == BoxShiftable)
	{
		boxFilter(src, dest, -1, kernelSize);
		minFilter(dest, dest, aggregationShiftableKernel);
	}
	else if (aggregationMethod == Gaussian)
	{
		GaussianBlur(src, dest, kernelSize, sigma_s_h, sigma_s_v);
	}
	else if (aggregationMethod == GaussianShiftable)
	{
		GaussianBlur(src, dest, kernelSize, sigma_s_h, sigma_s_v);
		minFilter(dest, dest, aggregationShiftableKernel);
	}
	else if (aggregationMethod == Guided)
	{
		//guidedImageFilter(src, guide, dest, max(aggregationRadiusH, aggregationRadiusV), aggregationGuidedfilterEps * aggregationGuidedfilterEps,GuidedTypes::GUIDED_SEP_VHI, BoxTypes::BOX_OPENCV, ParallelTypes::NAIVE);
		//guidedImageFilter(src, guide, dest, max(aggregationRadiusH, aggregationRadiusV), aggregationGuidedfilterEps * aggregationGuidedfilterEps, GuidedTypes::GUIDED_SEP_VHI_SHARE, BoxTypes::BOX_OPENCV, ParallelTypes::NAIVE);
		gif[omp_get_thread_num()].filterGuidePrecomputed(src, guide, dest, min(aggregationRadiusH, aggregationRadiusV), float(aggregationGuidedfilterEps * aggregationGuidedfilterEps), GuidedTypes::GUIDED_SEP_VHI_SHARE, ParallelTypes::NAIVE);
		//guidedImageFilter(src, guide, dest, max(aggregationRadiusH, aggregationRadiusV), aggregationGuidedfilterEps * aggregationGuidedfilterEps, GuidedTypes::GUIDED_SEP_VHI_SHARE, BoxTypes::BOX_OPENCV, ParallelTypes::OMP);
	}
	else if (aggregationMethod == CrossBasedBox)
	{
		clf(src, dest);
	}
	else if (aggregationMethod == Bilateral)
	{
		jointBilateralFilter(src, guide, dest, min(2 * aggregationRadiusH + 1, 2 * aggregationRadiusV + 1), aggregationGuidedfilterEps, sigma_s_h, FILTER_RECTANGLE);
	}
}

//WTA and optimization
void StereoMatch::computeOptimizeScanline()
{
	cout << "opt scan\n";
	Size size = DSI[0].size();
	Mat disp;

	Mat costMap = Mat::ones(size, CV_8U) * 255;
	const int imsize = size.area();
	//DSI[numberOfDisparities] = Mat::ones(DSI[0].size(),DSI[0].type())*192;
	//int* cost = new int[numberOfDisparities+1];
	//int* vd = new int[size.width+1];
	int cost[100];
	int vd[1000];
	{
		int j = 0;
		int pd = 1;
		for (int i = 0; i < size.width; i++)
		{
			for (int n = 1; n < numberOfDisparities - 1; n++)
			{
				cost[n] = (DSI[n].at<uchar>(j, i) += P2);
			}
			cost[pd] = (DSI[pd].at<uchar>(j, i) -= P2);
			cost[pd + 1] = (DSI[pd + 1].at<uchar>(j, i) -= (P2 - P1));
			cost[pd - 1] = (DSI[pd - 1].at<uchar>(j, i) -= (P2 - P1));

			int maxc = 65535;
			for (int n = 0; n < numberOfDisparities; n++)
			{
				if (cost[n] < maxc)
				{
					maxc = cost[n];
					pd = n;
				}
			}
			pd = max(pd, 1);
			pd = min(pd, numberOfDisparities - 2);
			vd[i] = pd;
		}
	}
	for (int j = 1; j < size.height; j++)
	{
		int pd;;
		{
			int maxc = 65535;
			for (int n = 0; n < numberOfDisparities; n++)
			{
				if (DSI[n].at<uchar>(j, 0) < maxc)
				{
					maxc = DSI[n].at<uchar>(j, 0);
					pd = n;
				}
			}
		}
		for (int i = 1; i < size.width; i++)
		{
			//nt apd = min(max(((pd + vd[i-1] + vd[i] + vd[i+1])>>2),1),numberOfDisparities-1);
			//int apd = min(max(vd[i],1),numberOfDisparities-1);
			//int apd = min(max(((2*pd + 2*vd[i])>>2),1),numberOfDisparities-1);
			int apd = min(max(pd, 1), numberOfDisparities - 2);
			for (int n = 0; n < numberOfDisparities; n++)
			{
				cost[n] = (DSI[n].at<uchar>(j, i) += P2);
			}

			cost[apd] = (DSI[apd].at<uchar>(j, i) -= P2);
			cost[apd + 1] = (DSI[apd + 1].at<uchar>(j, i) -= (P2 - P1));
			cost[apd - 1] = (DSI[apd - 1].at<uchar>(j, i) -= (P2 - P1));

			int maxc = 65535;
			for (int n = 0; n < numberOfDisparities; n++)
			{
				if (cost[n] < maxc)
				{
					maxc = cost[n];
					pd = n;
				}
			}
			vd[i] = pd;
		}
	}
	//delete[] cost;
	//delete[] vd;
}

void StereoMatch::computeWTA(vector<Mat> &dsi, Mat & dest, Mat & minimumCostMap)
{
	const int imsize = dest.size().area();
	const int simdsize = get_simd_floor(imsize, 32);
#if 0
	for (int i = 0; i < numberOfDisparities; i++)
	{
		const short d = ((minDisparity + i) << 4);

		short* disp = dest.ptr<short>(0);
		uchar* pDSI = dsi[i].data;
		uchar* cost = minimumCostMap.data;

		const __m256i md = _mm256_set1_epi16(d);
		for (int j = simdsize; j -= 32; pDSI += 32, cost += 32, disp += 32)
		{
			__m256i mdsi = _mm256_load_si256((__m256i*) pDSI);
			__m256i mcost = _mm256_load_si256((__m256i*) cost);

			__m256i  mask = _mm256_cmpgt_epu8(mcost, mdsi);
			mcost = _mm256_blendv_epi8(mcost, mdsi, mask);
			_mm256_store_si256((__m256i*)cost, mcost);

			__m256i  mdisp = _mm256_load_si256((__m256i*) disp);
			_mm256_store_si256((__m256i*)disp, _mm256_blendv_epi8(mdisp, md, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(mask))));

			mdisp = _mm256_load_si256((__m256i*) (disp + 16));
			_mm256_store_si256((__m256i*)(disp + 16), _mm256_blendv_epi8(mdisp, md, _mm256_cvtepi8_epi16(_mm256_extractf128_si256(mask, 1))));
		}
	}
#else
	short* disparityMapPtr = dest.ptr<short>();
#pragma omp parallel for //schedule (dynamic)
	for (int i = 0; i < simdsize; i += 32)
	{
		__m256i mcost = _mm256_set1_epi8(255);
		__m256i mdisp1 = _mm256_setzero_si256();
		__m256i mdisp2 = _mm256_setzero_si256();
		for (int d = 0; d < numberOfDisparities; d++)
		{
			const short disp_val = ((minDisparity + d) << 4);
			uchar* pDSI = dsi[d].data;
			const __m256i md = _mm256_set1_epi16(disp_val);
			__m256i mdsi = _mm256_load_si256((__m256i*) (pDSI + i));
			__m256i  mask = _mm256_cmpgt_epu8(mcost, mdsi);
			mcost = _mm256_blendv_epi8(mcost, mdsi, mask);
			mdisp1 = _mm256_blendv_epi8(mdisp1, md, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(mask)));
			mdisp2 = _mm256_blendv_epi8(mdisp2, md, _mm256_cvtepi8_epi16(_mm256_extractf128_si256(mask, 1)));
		}
		uchar* cost = minimumCostMap.data;
		_mm256_store_si256((__m256i*)(cost + i), mcost);
		_mm256_store_si256((__m256i*)(disparityMapPtr + i), mdisp1);
		_mm256_store_si256((__m256i*)(disparityMapPtr + i + 16), mdisp2);
	}
	for (int i = simdsize; i < imsize; i++)
	{
		uchar mcost = 255;
		short mind = 0;

		for (int d = 0; d < numberOfDisparities; d++)
		{
			const short disp_val = ((minDisparity + d) << 4);
			uchar* pDSI = dsi[d].data;
			if (pDSI[i] < mcost)
			{
				mcost = pDSI[i];
				mind = d;
			}
		}
		uchar* cost = minimumCostMap.data;
		cost[i] = mcost;
		disparityMapPtr[i] = mind;
	}
#endif
	}
#pragma endregion

#pragma region post filter
//post filter
void StereoMatch::uniquenessFilter(Mat & minCostMap, Mat & dest)
{
	if (uniquenessRatio == 0)return;
	if (!isUniquenessFilter) return;

	const int imsize = dest.size().area();
	const int simdsize = get_simd_floor(imsize, 16);

	const float mul = 1.f + uniquenessRatio / 100.f;
	const __m256i mmul = _mm256_set1_epi16((short)((mul - 1.f) * pow(2, 15)));

	uchar* mincostPtr = minCostMap.data;
	short* destPtr = dest.ptr<short>();

	for (int d = 0; d < numberOfDisparities; d++)
	{
		const short disparity = ((minDisparity + d) << 4);
		uchar* DSIPtr = DSI[d].data;

#if 0 //naive
		for (int i = 0; i < imsize; i++)
		{
			const int v = (mincostPtr[i] * mul);
			if ((*DSIPtr) < v && abs(disparity - destPtr[i]) > 16)
			{
				destPtr[i] = 0;//(minDisparity-1)<<4;
			}
		}
#else //SIMD
		const __m256i md = _mm256_set1_epi16(disparity);
		const __m256i m16 = _mm256_set1_epi16(16);
		for (int i = 0; i < simdsize; i += 16)
		{
			__m256i mc = _mm256_cvtepu8_epi16(_mm_load_si128((__m128i*)(mincostPtr + i)));
			__m256i mv = _mm256_add_epi16(mc, _mm256_mulhrs_epi16(mc, mmul));
			__m256i mdsi = _mm256_cvtepu8_epi16(_mm_load_si128((__m128i*)(DSIPtr + i)));
			__m256i mdest = _mm256_load_si256((__m256i*)(destPtr + i));

			__m256i mask1 = _mm256_cmpgt_epi16(mv, mdsi);
			__m256i mask2 = _mm256_cmpgt_epi16(_mm256_abs_epi16(_mm256_sub_epi16(md, mdest)), m16);
			mask1 = _mm256_and_si256(mask1, mask2);

			_mm256_store_si256((__m256i*)(destPtr + i), _mm256_blendv_epi8(mdest, _mm256_setzero_si256(), mask1));
		}
		for (int i = simdsize; i < imsize; i++)
		{
			int v = saturate_cast<int>(mincostPtr[i] * mul);
			if ((*DSIPtr) < v && abs(disparity - destPtr[i]) > 16)
			{
				destPtr[i] = 0;//(minDisparity-1)<<4;
			}
		}
#endif
		}
	}


string StereoMatch::getSubpixelInterpolationMethodName(const SUBPIXEL method)
{
	string mes;
	switch (method)
	{
	case SUBPIXEL::NONE:	mes = "Subpixel None"; break;
	case SUBPIXEL::QUAD:	mes = "Subpixel Quad"; break;
	case SUBPIXEL::LINEAR:	mes = "Subpixel Linear"; break;

	default:				mes = "This subpixel interpolation method is not supported"; break;
	}
	return mes;
}

void StereoMatch::setSubpixelInterpolationMethodName(const SUBPIXEL method)
{
	subpixelInterpolationMethod = (int)method;
}

void StereoMatch::subpixelInterpolation(Mat & disparity16, const SUBPIXEL method)
{
	if (method == SUBPIXEL::NONE) return;

	short* disp = disparity16.ptr<short>();
	const int imsize = disparity16.size().area();
	if (method == SUBPIXEL::QUAD)
	{
#pragma omp parallel for
		for (int j = 0; j < imsize; j++)
		{
			const short d = disp[j] >> 4;
			const int l = d - minDisparity;
			if (0 < l && l < numberOfDisparities - 1)
			{
				int f = DSI[l].data[j];
				int p = DSI[l + 1].data[j];
				int m = DSI[l - 1].data[j];

				int md = ((p + m - (f << 1)) << 1);
				if (md != 0)
				{
					const float dd = (float)d + (float)(m - p) / (float)md;
					disp[j] = saturate_cast<short>(16.f * dd);
				}
			}
		}
	}
	else if (method == SUBPIXEL::LINEAR)
	{
		for (int j = 0; j < imsize; j++)
		{
			const short d = disp[j] >> 4;
			const int l = d - minDisparity;
			if (0 < l && l < numberOfDisparities - 1)
			{
				const float f = (float)DSI[l].data[j];
				const float p = (float)DSI[l + 1].data[j];
				const float m = (float)DSI[l - 1].data[j];
				const float m31 = p - f;
				const float m21 = m - f;
				float md;
				if (m >= p)
				{
					//md = 0.5f*(m - p) / (m - f);
					md = 0.5f - 0.25f * ((m31 * m31) / (m21 * m21) + m31 / m21);
				}
				else
				{
					//md = 0.5f*(p - m) / (p - f);
					md = -(0.5f - 0.25f * ((m21 * m21) / (m31 * m31) + m21 / m31));
				}
				//if (md < 0.f || md>1.f)cout << md << endl;
				disp[j] = saturate_cast<short>(16.f * (float)(d + md));
			}
		}
	}
}


void StereoMatch::fastLRCheck(Mat & dest)
{
	Mat dispR = Mat::zeros(dest.size(), CV_16S);
	Mat disp8(dest.size(), CV_16S);
	const int imsize = dest.size().area();

	dest.convertTo(disp8, CV_16S, 1.0 / 16.0);

	short* displ = disp8.ptr<short>();
	short* dispr = dispR.ptr<short>();
	displ += minDisparity + numberOfDisparities;
	dispr += minDisparity + numberOfDisparities;
	for (int j = imsize - (minDisparity + numberOfDisparities); j--; displ++, dispr++)
	{
		const short d = *displ;
		if (d != 0)
		{
			if (*(dispr - d) < d)
			{
				*(dispr - d) = d;
			}
		}
	}
	LRCheckDisparity(disp8, dispR, 0, disp12diff + 1, 0, 1, LR_CHECK_DISPARITY_ONLY_L);
	//LRCheckDisparity(disp8,dispR,0,disp12diff,0,1,xcv::LR_CHECK_DISPARITY_BOTH);
	//Mat dr; dispR.convertTo(dr,CV_8U,4);imshow("dispR",dr);

	Mat mask;
	compare(disp8, 0, mask, cv::CMP_EQ);
	dest.setTo(0, mask);
}

void StereoMatch::fastLRCheck(Mat & costMap, Mat & srcdest)
{
	Mat dispR = Mat::zeros(srcdest.size(), CV_16S);
	Mat disp8(srcdest.size(), CV_16S);
	//dest.convertTo(disp8,CV_16S,1.0/16,0.5);
	const int imsize = srcdest.size().area();

	srcdest.convertTo(disp8, CV_16S, 1.0 / 16.0);
	if (isProcessLBorder)
	{
		const int disparity_max = minDisparity + numberOfDisparities;
		for (int j = 0; j < disp8.rows; j++)
		{
			short* dst = disp8.ptr<short>(j);
			memset(dst, 0, sizeof(short) * minDisparity + 1);
			for (int i = minDisparity + 1; i < disparity_max; i++)
			{
				dst[i] = (dst[i] >= i) ? 0 : dst[i];
			}
		}
	}

	Mat costMapR(srcdest.size(), CV_8U);
	costMapR.setTo(255);

	short* disp = disp8.ptr<short>(0);
	short* dispr = dispR.ptr<short>(0);

	uchar* cost = costMap.data;
	uchar* costr = costMapR.data;

	cost += minDisparity + numberOfDisparities;
	costr += minDisparity + numberOfDisparities;
	disp += minDisparity + numberOfDisparities;
	dispr += minDisparity + numberOfDisparities;
	for (int j = imsize - (minDisparity + numberOfDisparities); j--; cost++, costr++, disp++, dispr++)
	{
		const short d = *disp;
		if (d != 0)
		{
			if (*(cost) < *(costr - d))
			{
				*(costr - d) = *cost;
				*(dispr - d) = d;
			}
		}
	}
	LRCheckDisparity(disp8, dispR, 0, disp12diff, 0, 1, LR_CHECK_DISPARITY_ONLY_L);

	if (isProcessLBorder)
	{
		const int disparity_max = minDisparity + numberOfDisparities;
		for (int j = 0; j < disp8.rows; j++)
		{
			short* dst = disp8.ptr<short>(j);
			for (int i = minDisparity + 1; i < disparity_max; i++)
			{
				short d = dst[i];
				if (d != 0)
				{
					if (dst[i + d] == 0)dst[i] = 0;
				}
			}
		}
	}
	//LRCheckDisparity(disp8,dispR,0,disp12diff,0,1,xcv::LR_CHECK_DISPARITY_BOTH);
	//Mat dr; dispR.convertTo(dr,CV_8U,4);imshow("dispR",dr);

	Mat mask;
	compare(disp8, 0, mask, cv::CMP_EQ);
	srcdest.setTo(0, mask);
}

std::string StereoMatch::getLRCheckMethodName(const LRCHECK method)
{
	string ret = "no support in LRCheckMethod";
	switch (method)
	{
	case LRCHECK::NONE:				ret = "NONE"; break;
	case LRCHECK::WITH_MINCOST:		ret = "WITH_MINCOST"; break;
	case LRCHECK::WITHOUT_MINCOST:	ret = "WITHOUT_MINCOST"; break;
	default:
		break;
	}
	return ret;
}

void StereoMatch::setLRCheckMethod(const LRCHECK method)
{
	LRCheckMethod = (int)method;
}


void StereoMatch::minCostSwapFilter(const Mat & minCostMap, Mat & destDisparity)
{
	Mat disptemp = destDisparity.clone();
	const int step = destDisparity.cols;

	for (int j = 0; j < minCostMap.rows; j++)
	{
		short* tempdisp = disptemp.ptr<short>(j);
		short* destdisp = destDisparity.ptr<short>(j);
		const uchar* costPtr = minCostMap.ptr<uchar>(j);
		for (int i = 1; i < minCostMap.cols; i++)
		{
			if (destdisp[0] != 0 && destdisp[-1] != 0 && abs(destdisp[-1] - destdisp[0]) >= 16)
			{
				if (costPtr[-1] < costPtr[0])
				{
					destdisp[0] = tempdisp[-1];
				}
				else
				{
					destdisp[-1] = tempdisp[0];
				}
			}

			destdisp++;
			tempdisp++;
			costPtr++;
		}
	}
}

void StereoMatch::minCostThresholdFilter(const Mat & minCostMap, Mat & destDisparity, const uchar threshold)
{
	const int size = minCostMap.size().area();
	const uchar* cost = minCostMap.ptr<uchar>();
	short* destdisp = destDisparity.ptr<short>();
	for (int i = 0; i < size; i++)
	{
		*destdisp = (*cost > threshold) ? 0 : *destdisp;
		destdisp++; cost++;
	}
}

string StereoMatch::getHollFillingMethodName(const HOLE_FILL method)
{
	string ret = "not support getHollFillingMethod";
	switch (method)
	{
	case NONE:					ret = "NONE"; break;
	case NEAREST_MIN_SCANLINE:	ret = "NEAREST_MIN_SCANLINE"; break;
	case METHOD2:				ret = "Method 2(under debug)"; break;
	case METHOD3:				ret = "Method 3(under debug)"; break;
	case METHOD4:				ret = "Method 4(under debug)"; break;
	default:
		break;
	}
	return ret;
}

void StereoMatch::computeValidRatio(const Mat & disparityMap)
{
	valid_ratio = 100.0 * countNonZero(disparityMap) / disparityMap.size().area();
}

void StereoMatch::setHoleFiillingMethodName(const HOLE_FILL method)
{
	holeFillingMethod = method;
}

double StereoMatch::getValidRatio()
{
	return valid_ratio;
}


std::string StereoMatch::getRefinementMethodName(const REFINEMENT method)
{
	string ret = "no support RefinementMethod";
	switch (method)
	{
	case REFINEMENT::NONE:				ret = "NONE"; break;
	case REFINEMENT::GIF_JNF:			ret = "GIF_JNF"; break;
	case REFINEMENT::WGIF_GAUSS_JNF:	ret = "WGIF_GAUSS_JNF"; break;
	case REFINEMENT::WGIF_BFSUB_JNF:	ret = "WGIF_BFSUB_JNF"; break;
	case REFINEMENT::WGIF_BFW_JNF:		ret = "WGIF_BFW_JNF"; break;
	case REFINEMENT::WGIF_DUALBFW_JNF:	ret = "WGIF_DUALBFW_JNF"; break;
	case REFINEMENT::JBF_JNF:			ret = "JBF_JNF"; break;
	case REFINEMENT::WJBF_GAUSS_JNF:	ret = "WJBF_GAUSS_JNF"; break;
	case REFINEMENT::WMF:				ret = "WMF"; break;
	case REFINEMENT::WWMF_GAUSS:		ret = "WWMF_GAUSS"; break;
	default:
		break;
	}
	return ret;
}

void StereoMatch::setRefinementMethod(const REFINEMENT method, const int refinementR, const float refinementSigmaRange, const float refinementSigmaSpace, const int jointNearestR)
{
	this->refinementMethod = (int)method;

	if (refinementR >= 0) this->refinementR = refinementR;
	if (refinementSigmaRange >= 0.f)this->refinementSigmaRange = refinementSigmaRange;
	if (refinementSigmaSpace >= 0.f)this->refinementSigmaSpace = refinementSigmaSpace;
	if (jointNearestR >= 0)this->jointNearestR = jointNearestR;
}


void StereoMatch::getWeightUniqness(Mat & disp)
{
	const int imsize = disp.size().area();
	Mat rmap = Mat::ones(disp.size(), CV_8U) * 255;
	for (int i = 0; i < numberOfDisparities; i++)
	{
		const short d = ((minDisparity + i) << 4);
		short* dis = disp.ptr<short>(0);
		uchar* pDSI = DSI[i].data;
		uchar* cost = minCostMap.data;
		uchar* r = rmap.data;
		for (int j = imsize; j--; pDSI++, cost++, dis++)
		{
			short dd = (*dis);
			int v = 1000 * (*cost);
			int u = (*pDSI) * (1000 - uniquenessRatio);
			if (u - v < 0 && abs(d - dd)>16)
			{
				//int vv = (abs((double)*cost-*pDSI)/(double)(error_truncate))*255.0;
				//cout<<abs((double)*cost-(double)*pDSI)<<","<<(abs((double)*cost-(double)*pDSI)/(double)(error_truncate))*255.0<<endl;
				if (*cost == *pDSI)*r = 0;
				//*r=min(vv,(int)*r);
			}

			r++;
		}
	}
	rmap.convertTo(weightMap, CV_32F, 1.0 / 255);
	Mat rshow;	applyColorMap(rmap, rshow, 2); imshow("rmap", rshow);
}

template <class srcType>
void singleDisparityLRCheck_(Mat & dest, double amp, int thresh, int minDisparity, int numberOfDisparities)
{
	const int imsize = dest.size().area();
	Mat dispR = Mat::zeros(dest.size(), dest.type());
	Mat disp8(dest.size(), dest.type());


	srcType* dddd = dest.ptr<srcType>(0);
	srcType* d8 = disp8.ptr<srcType>(0);

	const double div = 1.0 / amp;
	if (amp != 1.0)
	{
		for (int i = 0; i < dest.size().area(); i++)
		{
			d8[i] = (srcType)(dddd[i] * div + 0.5);
		}
	}
	else
		dest.copyTo(disp8);

	srcType* disp = disp8.ptr<srcType>(0);
	srcType* dispr = dispR.ptr<srcType>(0);

	disp += minDisparity + numberOfDisparities;
	dispr += minDisparity + numberOfDisparities;
	for (int j = imsize - (minDisparity + numberOfDisparities); j--; disp++, dispr++)
	{
		const short d = *disp;
		if (d != 0)
		{
			if (*(dispr - d) < *(disp))
			{
				*(dispr - d) = (srcType)d;
			}
		}
	}
	LRCheckDisparity(disp8, dispR, minDisparity + numberOfDisparities, thresh + 1, 0, 1, LR_CHECK_DISPARITY_ONLY_L);
	//LRCheckDisparity(disp8,dispR,0,thresh,0,1,xcv::LR_CHECK_DISPARITY_BOTH);
	//Mat dr; dispR.convertTo(dr,CV_8U,4);imshow("dispR",dr);

	Mat mask;
	compare(disp8, 0, mask, cv::CMP_EQ);
	dest.setTo(0, mask);

}

void singleDisparityLRCheck(Mat & dest, double amp, int thresh, int minDisparity, int numberOfDisparities)
{
	if (dest.type() == CV_8U)
		singleDisparityLRCheck_<uchar>(dest, amp, thresh, minDisparity, numberOfDisparities);
	if (dest.type() == CV_16S)
		singleDisparityLRCheck_<short>(dest, amp, thresh, minDisparity, numberOfDisparities);
	if (dest.type() == CV_16U)
		singleDisparityLRCheck_<unsigned short>(dest, amp, thresh, minDisparity, numberOfDisparities);
}
#pragma endregion