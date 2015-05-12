#pragma once

#define WK			3
//# define THREAD_AMOUNT		4
#define GATHER_FEATURES
#define HSV
#define TRAIN
#define CLASSIFY

#ifdef GATHER_FEATURES
typedef struct 
{
	int File_Id,

		C1_EnergyMean, C1_EnergyVariance,
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
		C3_HaralickCorrelationMean, C3_HaralickCorrelationVariance,
		
		Class;
} FeaturePrinterIndexStruct;

#define IMAGES_DIR	"resources\\images\\"
#endif

#define TRAIN_DIR "resources\\model\\"

#ifdef CLASSIFY
#define CLASSIFY_DIR	"resources\\classification\\"
#endif