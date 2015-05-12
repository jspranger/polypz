#pragma once

#include <opencv2\ml\ml.hpp>

class FeatureClassifier
{
public:
	FeatureClassifier(std::vector<std::string> features = std::vector<std::string>());
	~FeatureClassifier(void);

	void saveModel(std::string);
	void loadModel(std::string);

	void train(std::string);

	bool predict(std::vector<double> fvalues);

private:
	CvMLData mlData;
	cv::RandomTrees m_classifier;
	//cv::Boost m_classifier;
	//cv::NormalBayesClassifier m_classifier;
};

