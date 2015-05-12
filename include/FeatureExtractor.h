#pragma once

#include <vector>
#include <cmath>
#include <opencv2\opencv.hpp>
#include "itkImage.h"
#include "itkOpenCVImageBridge.h"
#include "itkScalarImageToCooccurrenceMatrixFilter.h"
#include "itkHistogramToTextureFeaturesFilter.h"

#include "..\include\fwt.h"

#include <limits>

#include "config.h"

typedef struct
{
	double Energy,
		   Entropy,
		   Correlation,
		   DifferenceMoment,
		   Inertia,
		   ClusterShade,
		   ClusterProminence,
		   HaralickCorrelation;
} Features;

typedef struct
{
	double C1_EnergyMean, C1_EnergyVariance,
		   C2_EnergyMean, C2_EnergyVariance,
		   C3_EnergyMean, C3_EnergyVariance,

		   C1_EntropyMean, C1_EntropyVariance,
		   C2_EntropyMean, C2_EntropyVariance,
		   C3_EntropyMean, C3_EntropyVariance,

		   C1_CorrelationMean, C1_CorrelationVariance,
		   C2_CorrelationMean, C2_CorrelationVariance,
		   C3_CorrelationMean, C3_CorrelationVariance,

		   C1_DifferenceMomentMean, C1_DifferenceMomentVariance,
		   C2_DifferenceMomentMean, C2_DifferenceMomentVariance,
		   C3_DifferenceMomentMean, C3_DifferenceMomentVariance,

		   C1_InertiaMean, C1_InertiaVariance,
		   C2_InertiaMean, C2_InertiaVariance,
		   C3_InertiaMean, C3_InertiaVariance,

		   C1_ClusterShadeMean, C1_ClusterShadeVariance,
		   C2_ClusterShadeMean, C2_ClusterShadeVariance,
		   C3_ClusterShadeMean, C3_ClusterShadeVariance,

		   C1_ClusterProminenceMean, C1_ClusterProminenceVariance,
		   C2_ClusterProminenceMean, C2_ClusterProminenceVariance,
		   C3_ClusterProminenceMean, C3_ClusterProminenceVariance,

		   C1_HaralickCorrelationMean, C1_HaralickCorrelationVariance,
		   C2_HaralickCorrelationMean, C2_HaralickCorrelationVariance,
		   C3_HaralickCorrelationMean, C3_HaralickCorrelationVariance;
} FeatureMV;

enum Color
{
	blue = 0,
	green = 1,
	red = 2
};

using namespace std;

typedef itk::Image<double, 2> itkImage;
typedef itk::Statistics::ScalarImageToCooccurrenceMatrixFilter<itkImage> itkCoocurrenceMatrixCalculator;
typedef itkCoocurrenceMatrixCalculator::HistogramType itkCoocurrenceMatrix;
typedef itk::Statistics::HistogramToTextureFeaturesFilter<itkCoocurrenceMatrix> itkTextureFeatureCalculator;

// Debug
#include "itkHistogramToIntensityImageFilter.h"
typedef itk::HistogramToIntensityImageFilter<itkCoocurrenceMatrix, itkImage> itkHistogramDebugger;

class FeatureExtractor
{
public:
	FeatureExtractor(string imagePath = "",
					 int waveletLevels = 3,
					 vector<itk::Offset<2> > offsets = vector<itk::Offset<2> >());
	~FeatureExtractor(void);

	Features F(Color c, int a);
	FeatureMV FMV(void);

private:
	bool m_status;

	//void adaptiveMedianBlur(cv::InputArray input, cv::OutputArray output, int ksize, int smax);
	static inline double round(double val)
	{    
		return floor(val + 0.5);
	}
	static inline double trunc(double val)
	{
		return (val > 0) ? floor(val) : ceil(val);
	}

	// Color Selection
	//   Level Selection
	//     Band Selection
	//       Angle Selection
	//         Features
	vector<vector<Features> > m_selection;

	// Wavelet
	fwt *m_fwt;
	int K;

	// Co-ocurrence statistical information angles
	vector<itk::Offset<2> > m_offsets;
};

