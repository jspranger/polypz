#include <Windows.h>

#include <stdio.h>
#include <tchar.h>
#include <regex>

#include <boost\thread.hpp>
#include <opencv2\ml\ml.hpp>

#include "config.h"


using namespace std;

vector<string> getFilelist(const char *searchkey)
{
	vector<string> m_filelist;

    WIN32_FIND_DATAA fd;
	HANDLE h;
	h = FindFirstFileA(searchkey, &fd);
	while (h != INVALID_HANDLE_VALUE)
	{
		m_filelist.push_back((string)(char *)fd.cFileName);
		if (!FindNextFileA(h, &fd))
			h = INVALID_HANDLE_VALUE;
	}
    return m_filelist;
}

#include "..\include\FeatureExtractor.h"
#include "..\include\FeaturePrinter.h"
#include "..\include\FeatureClassifier.h"

#ifdef GATHER_FEATURES
#ifdef HSV
FeaturePrinter m_featurePrinter("resources\\model\\featuresHSV.arff");
#else
FeaturePrinter m_featurePrinter("resources\\model\\featuresRGB.arff");
#endif
#endif

FeatureClassifier m_featureClassifier;

#ifdef GATHER_FEATURES
FeaturePrinterIndexStruct FeaturePrinterIndex;
int feature_counter = 0;
#endif

void workerFeatureExtraction(string path,
							 string file,
							 vector<itk::Offset<2> > *offsets,
							 vector<int> *gTruth,
							 FeaturePrinter *featurePrinter)
{
	regex m_regex("(\\d+).*");
	smatch m_match;
	if (regex_match(file, m_match, m_regex))
	{
		FeatureExtractor m_featureExtractor(path + file,
											WK,
											*offsets);

		FeatureMV m_features = m_featureExtractor.FMV();

#ifdef GATHER_FEATURES

		int m_fileID = atoi(m_match[1].str().c_str()) - 1;

		featurePrinter->addData(FeaturePrinterIndex.File_Id, m_fileID + 1);
		featurePrinter->addData(FeaturePrinterIndex.C1_EnergyMean, m_features.C1_EnergyMean);
		featurePrinter->addData(FeaturePrinterIndex.C1_EnergyVariance, m_features.C1_EnergyVariance);
		featurePrinter->addData(FeaturePrinterIndex.C2_EnergyMean, m_features.C2_EnergyMean);
		featurePrinter->addData(FeaturePrinterIndex.C2_EnergyVariance, m_features.C2_EnergyVariance);
		featurePrinter->addData(FeaturePrinterIndex.C3_EnergyMean, m_features.C3_EnergyMean);
		featurePrinter->addData(FeaturePrinterIndex.C3_EnergyVariance, m_features.C3_EnergyVariance);

		featurePrinter->addData(FeaturePrinterIndex.C1_EntropyMean, m_features.C1_EntropyMean);
		featurePrinter->addData(FeaturePrinterIndex.C1_EntropyVariance, m_features.C1_EntropyVariance);
		featurePrinter->addData(FeaturePrinterIndex.C2_EntropyMean, m_features.C2_EntropyMean);
		featurePrinter->addData(FeaturePrinterIndex.C2_EntropyVariance, m_features.C2_EntropyVariance);
		featurePrinter->addData(FeaturePrinterIndex.C3_EntropyMean, m_features.C3_EntropyMean);
		featurePrinter->addData(FeaturePrinterIndex.C3_EntropyVariance, m_features.C3_EntropyVariance);

		featurePrinter->addData(FeaturePrinterIndex.C1_CorrelationMean, m_features.C1_CorrelationMean);
		featurePrinter->addData(FeaturePrinterIndex.C1_CorrelationVariance, m_features.C1_CorrelationVariance);
		featurePrinter->addData(FeaturePrinterIndex.C2_CorrelationMean, m_features.C2_CorrelationMean);
		featurePrinter->addData(FeaturePrinterIndex.C2_CorrelationVariance, m_features.C2_CorrelationVariance);
		featurePrinter->addData(FeaturePrinterIndex.C3_CorrelationMean, m_features.C3_CorrelationMean);
		featurePrinter->addData(FeaturePrinterIndex.C3_CorrelationVariance, m_features.C3_CorrelationVariance);

		featurePrinter->addData(FeaturePrinterIndex.C1_DifferenceMomentMean, m_features.C1_DifferenceMomentMean);
		featurePrinter->addData(FeaturePrinterIndex.C1_DifferenceMomentVariance, m_features.C1_DifferenceMomentVariance);
		featurePrinter->addData(FeaturePrinterIndex.C2_DifferenceMomentMean, m_features.C2_DifferenceMomentMean);
		featurePrinter->addData(FeaturePrinterIndex.C2_DifferenceMomentVariance, m_features.C2_DifferenceMomentVariance);
		featurePrinter->addData(FeaturePrinterIndex.C3_DifferenceMomentMean, m_features.C3_DifferenceMomentMean);
		featurePrinter->addData(FeaturePrinterIndex.C3_DifferenceMomentVariance, m_features.C3_DifferenceMomentVariance);

		featurePrinter->addData(FeaturePrinterIndex.C1_InertiaMean, m_features.C1_InertiaMean);
		featurePrinter->addData(FeaturePrinterIndex.C1_InertiaVariance, m_features.C1_InertiaVariance);
		featurePrinter->addData(FeaturePrinterIndex.C2_InertiaMean, m_features.C2_InertiaMean);
		featurePrinter->addData(FeaturePrinterIndex.C2_InertiaVariance, m_features.C2_InertiaVariance);
		featurePrinter->addData(FeaturePrinterIndex.C3_InertiaMean, m_features.C3_InertiaMean);
		featurePrinter->addData(FeaturePrinterIndex.C3_InertiaVariance, m_features.C3_InertiaVariance);

		featurePrinter->addData(FeaturePrinterIndex.C1_ClusterShadeMean, m_features.C1_ClusterShadeMean);
		featurePrinter->addData(FeaturePrinterIndex.C1_ClusterShadeVariance, m_features.C1_ClusterShadeVariance);
		featurePrinter->addData(FeaturePrinterIndex.C2_ClusterShadeMean, m_features.C2_ClusterShadeMean);
		featurePrinter->addData(FeaturePrinterIndex.C2_ClusterShadeVariance, m_features.C2_ClusterShadeVariance);
		featurePrinter->addData(FeaturePrinterIndex.C3_ClusterShadeMean, m_features.C3_ClusterShadeMean);
		featurePrinter->addData(FeaturePrinterIndex.C3_ClusterShadeVariance, m_features.C3_ClusterShadeVariance);

		featurePrinter->addData(FeaturePrinterIndex.C1_ClusterProminenceMean, m_features.C1_ClusterProminenceMean);
		featurePrinter->addData(FeaturePrinterIndex.C1_ClusterProminenceVariance, m_features.C1_ClusterProminenceVariance);
		featurePrinter->addData(FeaturePrinterIndex.C2_ClusterProminenceMean, m_features.C2_ClusterProminenceMean);
		featurePrinter->addData(FeaturePrinterIndex.C2_ClusterProminenceVariance, m_features.C2_ClusterProminenceVariance);
		featurePrinter->addData(FeaturePrinterIndex.C3_ClusterProminenceMean, m_features.C3_ClusterProminenceMean);
		featurePrinter->addData(FeaturePrinterIndex.C3_ClusterProminenceVariance, m_features.C3_ClusterProminenceVariance);

		featurePrinter->addData(FeaturePrinterIndex.C1_HaralickCorrelationMean, m_features.C1_HaralickCorrelationMean);
		featurePrinter->addData(FeaturePrinterIndex.C1_HaralickCorrelationVariance, m_features.C1_HaralickCorrelationVariance);
		featurePrinter->addData(FeaturePrinterIndex.C2_HaralickCorrelationMean, m_features.C2_HaralickCorrelationMean);
		featurePrinter->addData(FeaturePrinterIndex.C2_HaralickCorrelationVariance, m_features.C2_HaralickCorrelationVariance);
		featurePrinter->addData(FeaturePrinterIndex.C3_HaralickCorrelationMean, m_features.C3_HaralickCorrelationMean);
		featurePrinter->addData(FeaturePrinterIndex.C3_HaralickCorrelationVariance, m_features.C3_HaralickCorrelationVariance);

		featurePrinter->addData(FeaturePrinterIndex.Class, gTruth->at(feature_counter));

		featurePrinter->printData();

		feature_counter++;
#endif

#ifdef CLASSIFY
		cv::Mat m_image = cv::imread(path + file);
		std::vector<double> m_instance;

		m_instance.push_back(m_features.C1_EnergyMean);
		m_instance.push_back(m_features.C1_EnergyVariance);
		m_instance.push_back(m_features.C2_EnergyMean);
		m_instance.push_back(m_features.C2_EnergyVariance);
		m_instance.push_back(m_features.C3_EnergyMean);
		m_instance.push_back(m_features.C3_EnergyVariance);

		m_instance.push_back(m_features.C1_EntropyMean);
		m_instance.push_back(m_features.C1_EntropyVariance);
		m_instance.push_back(m_features.C2_EntropyMean);
		m_instance.push_back(m_features.C2_EntropyVariance);
		m_instance.push_back(m_features.C3_EntropyMean);
		m_instance.push_back(m_features.C3_EntropyVariance);

		m_instance.push_back(m_features.C1_DifferenceMomentMean);
		m_instance.push_back(m_features.C1_DifferenceMomentVariance);
		m_instance.push_back(m_features.C2_DifferenceMomentMean);
		m_instance.push_back(m_features.C2_DifferenceMomentVariance);
		m_instance.push_back(m_features.C3_DifferenceMomentMean);
		m_instance.push_back(m_features.C3_DifferenceMomentVariance);

		m_instance.push_back(m_features.C1_HaralickCorrelationMean);
		m_instance.push_back(m_features.C1_HaralickCorrelationVariance);
		m_instance.push_back(m_features.C2_HaralickCorrelationMean);
		m_instance.push_back(m_features.C2_HaralickCorrelationVariance);
		m_instance.push_back(m_features.C3_HaralickCorrelationMean);
		m_instance.push_back(m_features.C3_HaralickCorrelationVariance);

		bool m_result = m_featureClassifier.predict(m_instance);

		cv::putText(m_image,
					m_result?"True":"False",
					cv::Point(10, 30),
					cv::FONT_HERSHEY_PLAIN,
					2,
					m_result?cv::Scalar(0, 255, 0, 255):cv::Scalar(0, 0, 255, 255),
					3,
					8);
		cv::imshow(file, m_image);
		cv::waitKey();
		cv::destroyWindow(file);
#endif
	}
}

// ________________________________________________________________________

int _tmain(int argc, _TCHAR* argv[])
{
	// Coocurrence Matrix Angle Offset
	vector <itk::Offset<2> > m_offsets;
	itk::Offset<2> angle_0 = { 1, 0 },
				   angle_45 = { 1, 1 },
				   angle_90 = { 0, 1 },
				   angle_135 = { -1 , 1 };
	m_offsets.push_back(angle_0);
	m_offsets.push_back(angle_45);
	m_offsets.push_back(angle_90);
	m_offsets.push_back(angle_135);

	#ifdef GATHER_FEATURES
	// Ground truth data
	vector<int> m_groundTruth;
	ifstream m_istream;
	m_istream.open(TRAIN_DIR"\\gtruth.txt");
	if (m_istream.good())
	{
		string s;
		while (getline(m_istream, s))
			m_groundTruth.push_back(atoi(s.c_str()));
	}

	if (m_featurePrinter.initRelation("polyps"))
	{
		FeaturePrinterIndex.File_Id = m_featurePrinter.initAttribute("FILE_ID", "NUMERIC");

		FeaturePrinterIndex.C1_EnergyMean = m_featurePrinter.initAttribute("C1ENERGYMEAN", "REAL");
		FeaturePrinterIndex.C1_EnergyVariance = m_featurePrinter.initAttribute("C1ENERGYVARIANCE", "REAL");
		FeaturePrinterIndex.C2_EnergyMean = m_featurePrinter.initAttribute("C2ENERGYMEAN", "REAL");
		FeaturePrinterIndex.C2_EnergyVariance = m_featurePrinter.initAttribute("C2ENERGYVARIANCE", "REAL");
		FeaturePrinterIndex.C3_EnergyMean = m_featurePrinter.initAttribute("C3ENERGYMEAN", "REAL");
		FeaturePrinterIndex.C3_EnergyVariance = m_featurePrinter.initAttribute("C3ENERGYVARIANCE", "REAL");

		FeaturePrinterIndex.C1_EntropyMean = m_featurePrinter.initAttribute("C1ENTROPYMEAN", "REAL");
		FeaturePrinterIndex.C1_EntropyVariance = m_featurePrinter.initAttribute("C1ENTROPYVARIANCE", "REAL");
		FeaturePrinterIndex.C2_EntropyMean = m_featurePrinter.initAttribute("C2ENTROPYMEAN", "REAL");
		FeaturePrinterIndex.C2_EntropyVariance = m_featurePrinter.initAttribute("C2ENTROPYVARIANCE", "REAL");
		FeaturePrinterIndex.C3_EntropyMean = m_featurePrinter.initAttribute("C3ENTROPYMEAN", "REAL");
		FeaturePrinterIndex.C3_EntropyVariance = m_featurePrinter.initAttribute("C3ENTROPYVARIANCE", "REAL");

		FeaturePrinterIndex.C1_CorrelationMean = m_featurePrinter.initAttribute("C1CORRELATIONMEAN", "REAL");
		FeaturePrinterIndex.C1_CorrelationVariance = m_featurePrinter.initAttribute("C1CORRELATIONVARIANCE", "REAL");
		FeaturePrinterIndex.C2_CorrelationMean = m_featurePrinter.initAttribute("C2CORRELATIONMEAN", "REAL");
		FeaturePrinterIndex.C2_CorrelationVariance = m_featurePrinter.initAttribute("C2CORRELATIONVARIANCE", "REAL");
		FeaturePrinterIndex.C3_CorrelationMean = m_featurePrinter.initAttribute("C3CORRELATIONMEAN", "REAL");
		FeaturePrinterIndex.C3_CorrelationVariance = m_featurePrinter.initAttribute("C3CORRELATIONVARIANCE", "REAL");

		FeaturePrinterIndex.C1_DifferenceMomentMean = m_featurePrinter.initAttribute("C1DIFFERENCEMOMENTMEAN", "REAL");
		FeaturePrinterIndex.C1_DifferenceMomentVariance = m_featurePrinter.initAttribute("C1DIFFERENCEMOMENTVARIANCE", "REAL");
		FeaturePrinterIndex.C2_DifferenceMomentMean = m_featurePrinter.initAttribute("C2DIFFERENCEMOMENTMEAN", "REAL");
		FeaturePrinterIndex.C2_DifferenceMomentVariance = m_featurePrinter.initAttribute("C2DIFFERENCEMOMENTVARIANCE", "REAL");
		FeaturePrinterIndex.C3_DifferenceMomentMean = m_featurePrinter.initAttribute("C3DIFFERENCEMOMENTMEAN", "REAL");
		FeaturePrinterIndex.C3_DifferenceMomentVariance = m_featurePrinter.initAttribute("C3DIFFERENCEMOMENTVARIANCE", "REAL");

		FeaturePrinterIndex.C1_InertiaMean = m_featurePrinter.initAttribute("C1INERTIAMEAN", "REAL");
		FeaturePrinterIndex.C1_InertiaVariance = m_featurePrinter.initAttribute("C1INERTIAVARIANCE", "REAL");
		FeaturePrinterIndex.C2_InertiaMean = m_featurePrinter.initAttribute("C2INERTIAMEAN", "REAL");
		FeaturePrinterIndex.C2_InertiaVariance = m_featurePrinter.initAttribute("C2INERTIAVARIANCE", "REAL");
		FeaturePrinterIndex.C3_InertiaMean = m_featurePrinter.initAttribute("C3INERTIAMEAN", "REAL");
		FeaturePrinterIndex.C3_InertiaVariance = m_featurePrinter.initAttribute("C3INERTIAVARIANCE", "REAL");

		FeaturePrinterIndex.C1_ClusterShadeMean = m_featurePrinter.initAttribute("C1CLUSTERSHADEMEAN", "REAL");
		FeaturePrinterIndex.C1_ClusterShadeVariance = m_featurePrinter.initAttribute("C1CLUSTERSHADEVARIANCE", "REAL");
		FeaturePrinterIndex.C2_ClusterShadeMean = m_featurePrinter.initAttribute("C2CLUSTERSHADEMEAN", "REAL");
		FeaturePrinterIndex.C2_ClusterShadeVariance = m_featurePrinter.initAttribute("C2CLUSTERSHADEVARIANCE", "REAL");
		FeaturePrinterIndex.C3_ClusterShadeMean = m_featurePrinter.initAttribute("C3CLUSTERSHADEMEAN", "REAL");
		FeaturePrinterIndex.C3_ClusterShadeVariance = m_featurePrinter.initAttribute("C3CLUSTERSHADEVARIANCE", "REAL");

		FeaturePrinterIndex.C1_ClusterProminenceMean = m_featurePrinter.initAttribute("C1CLUSTERPROMINENCEMEAN", "REAL");
		FeaturePrinterIndex.C1_ClusterProminenceVariance = m_featurePrinter.initAttribute("C1CLUSTERPROMINENCEVARIANCE", "REAL");
		FeaturePrinterIndex.C2_ClusterProminenceMean = m_featurePrinter.initAttribute("C2CLUSTERPROMINENCEMEAN", "REAL");
		FeaturePrinterIndex.C2_ClusterProminenceVariance = m_featurePrinter.initAttribute("C2CLUSTERPROMINENCEVARIANCE", "REAL");
		FeaturePrinterIndex.C3_ClusterProminenceMean = m_featurePrinter.initAttribute("C3CLUSTERPROMINENCEMEAN", "REAL");
		FeaturePrinterIndex.C3_ClusterProminenceVariance = m_featurePrinter.initAttribute("C3CLUSTERPROMINENCEVARIANCE", "REAL");

		FeaturePrinterIndex.C1_HaralickCorrelationMean = m_featurePrinter.initAttribute("C1HARALICKCORRELATIONMEAN", "REAL");
		FeaturePrinterIndex.C1_HaralickCorrelationVariance = m_featurePrinter.initAttribute("C1HARALICKCORRELATIONVARIANCE", "REAL");
		FeaturePrinterIndex.C2_HaralickCorrelationMean = m_featurePrinter.initAttribute("C2HARALICKCORRELATIONMEAN", "REAL");
		FeaturePrinterIndex.C2_HaralickCorrelationVariance = m_featurePrinter.initAttribute("C2HARALICKCORRELATIONVARIANCE", "REAL");
		FeaturePrinterIndex.C3_HaralickCorrelationMean = m_featurePrinter.initAttribute("C3HARALICKCORRELATIONMEAN", "REAL");
		FeaturePrinterIndex.C3_HaralickCorrelationVariance = m_featurePrinter.initAttribute("C3HARALICKCORRELATIONVARIANCE", "REAL");

		FeaturePrinterIndex.Class = m_featurePrinter.initAttribute("CLASS", "{0,1}");
	}
	// Read all images
	//vector<boost::thread> threads;
	vector<string> m_filelist = getFilelist((const char *)(IMAGES_DIR"*"));
	for (int i = 0; i < m_filelist.size(); i++)
	{
		//threads.push_back(boost::thread(workerFeatureExtraction, m_filelist.at(i), &m_offsets, &m_groundTruth, &m_featurePrinter));
		//if ((threads.size() >= THREAD_AMOUNT) || (i >= m_filelist.size() - 1))
		//{
		//	for (int t = 0; t < threads.size(); t++)
		//		threads[t].join();
		//	threads.clear();
		//}
		workerFeatureExtraction(IMAGES_DIR, m_filelist.at(i), &m_offsets, &m_groundTruth, &m_featurePrinter);
	}
	#endif

#ifdef TRAIN
	#ifdef HSV
		m_featureClassifier.train(TRAIN_DIR"\\trainDataHSV.csv");
		m_featureClassifier.saveModel(TRAIN_DIR"\\trainModelHSV.xml");
	#else
		m_featureClassifier.train(TRAIN_DIR"\\trainDataRGB.csv");
		m_featureClassifier.saveModel(TRAIN_DIR"\\trainModelRGB.xml");
	#endif
#endif

#ifdef CLASSIFY
	// LOAD CLASSIFICATION MODEL
	#ifdef HSV
		m_featureClassifier.loadModel(TRAIN_DIR"\\trainModelHSV.xml");
	#else
		m_featureClassifier.loadModel(TRAIN_DIR"\\trainModelRGB.xml");
	#endif

	// Read all images
	//vector<boost::thread> threads;
	vector<string> m_filelist = getFilelist((const char *)(CLASSIFY_DIR"\\*"));
	for (int i = 0; i < m_filelist.size(); i++)
	{
		//threads.push_back(boost::thread(workerFeatureExtraction, m_filelist.at(i), &m_offsets, &m_groundTruth, &m_featurePrinter));
		//if ((threads.size() >= THREAD_AMOUNT) || (i >= m_filelist.size() - 1))
		//{
		//	for (int t = 0; t < threads.size(); t++)
		//		threads[t].join();
		//	threads.clear();
		//}
		workerFeatureExtraction(CLASSIFY_DIR, m_filelist.at(i), &m_offsets, 0, 0);
	}
#endif

	system("PAUSE");

	return 0;
}
