#include "..\include\FeatureClassifier.h"

#include <opencv2\opencv.hpp>

FeatureClassifier::FeatureClassifier(std::vector<std::string> features)
{
}

FeatureClassifier::~FeatureClassifier(void)
{
}

void FeatureClassifier::saveModel(std::string path)
{
	m_classifier.save(path.c_str());
}

void FeatureClassifier::loadModel(std::string path)
{
	m_classifier.load(path.c_str());
}

void FeatureClassifier::train(std::string path)
{
	mlData.read_csv(path.c_str());

	// Class index
	mlData.set_response_idx(mlData.get_values()->cols - 1);
	mlData.change_var_type(mlData.get_values()->cols - 1, CV_VAR_CATEGORICAL);

	// Slit train and test data
	CvTrainTestSplit m_trainTestSplit(0.66f, true);
	mlData.set_train_test_split(&m_trainTestSplit);

	cv::Mat m_dataset = mlData.get_values();

	cv::Mat m_values,
			m_responses = mlData.get_responses(),

			m_trainIndex = mlData.get_train_sample_idx(),
			m_trainData,
			m_trainResponse,

			m_testIndex = mlData.get_test_sample_idx(),
			m_testData,
			m_testResponse;

	m_dataset.colRange(0, m_dataset.cols - 1).copyTo(m_values);

	for (int i = 0; i < m_trainIndex.cols; i++)
	{
		m_trainData.push_back(m_values.row(m_trainIndex.at<int>(i)));
		m_trainResponse.push_back(m_responses.row(m_trainIndex.at<int>(i)));
	}

	for (int i = 0; i < (m_testIndex.cols); i++)
	{
		m_testData.push_back(m_values.row(m_testIndex.at<int>(i)));
		m_testResponse.push_back(m_responses.row(m_testIndex.at<int>(i)));
	}

	// TRAIN MODEL
	m_classifier.train(m_trainData,
					   CV_ROW_SAMPLE,
					   m_trainResponse,
					   cv::Mat(),
					   cv::Mat(),
					   cv::Mat(),
					   cv::Mat(),
					   CvRTParams(10,
								  5,
								  0,
								  false,
								  2,
								  0,
								  false,
								  0,
								  150,
								  0.01f,
								  CV_TERMCRIT_ITER | CV_TERMCRIT_EPS)); // Random Trees

	//m_classifier.train(m_trainData, m_trainResponse); // Bayes
	//m_classifier.train(m_trainData,
	//				   CV_ROW_SAMPLE,
	//				   m_trainResponse,
	//				   cv::Mat(),
	//				   cv::Mat(),
	//				   cv::Mat(),
	//				   cv::Mat(),
	//				   cv::BoostParams(CvBoost::REAL, 100, 0.95, 1, false, 0),
	//				   false); // Boost

	// TEST MODEL
	int m_confusionMatrix[2][2];
	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 2; j++)
			m_confusionMatrix[i][j] = 0;
	for (int i = 0; i < m_testIndex.cols; i++)
	{
		int m_realLabel = ((int)m_testResponse.at<float>(i, 0)) - 1;
		int m_predictedLabel = (m_classifier.predict_prob(m_testData.row(i)) >= 0.5f); // RandomForest
		//int m_predictedLabel = m_classifier.predict(m_testData.row(i)) - 1; // Bayes and Boost

		if ((m_realLabel == 1) && (m_predictedLabel == 1))
			m_confusionMatrix[0][0] += 1; // TP

		if ((m_realLabel == 1) && (m_predictedLabel == 0))
			m_confusionMatrix[1][0] += 1; // FN

		if ((m_realLabel == 0) && (m_predictedLabel == 1))
			m_confusionMatrix[0][1] += 1; // FP

		if ((m_realLabel == 0) && (m_predictedLabel == 0))
			m_confusionMatrix[1][1] += 1; // TN
	}

	float TPR = m_confusionMatrix[0][0] / (float)(m_confusionMatrix[0][0] + m_confusionMatrix[0][1]),
		  FPR = m_confusionMatrix[1][1] / (float)(m_confusionMatrix[1][0] + m_confusionMatrix[1][1]);

	std::cout << "Confusion Matrix _________" << std::endl;
	std::cout << m_confusionMatrix[0][0] << " | " << m_confusionMatrix[1][0] << std::endl;
	std::cout << m_confusionMatrix[0][1] << " | " << m_confusionMatrix[1][1] << std::endl << std::endl;

	std::cout << "Sensitivity = " << TPR << std::endl;
	std::cout << "1 - Specificity = " << 1 - FPR << std::endl << std::endl;
}

bool FeatureClassifier::predict(std::vector<double> instance)
{
	cv::Mat m_data = cv::Mat::zeros(1, instance.size(), CV_32FC1);
	for (int i = 0; i < instance.size(); i++)
		m_data.at<float>(i) = (float)instance.at(i);

	float result = m_classifier.predict_prob(m_data); // RandomForest
	//float result = m_classifier.predict(m_data); // Bayes and Boost

	return (result >= 0.5f);
}
