#pragma once

#include <opencv2\opencv.hpp>
#include <vector>

using namespace std;

class fwt
{
public:
	typedef enum
	{
		Haar,
	} Type;

	fwt(fwt::Type type = fwt::Haar, int levels = 3);
	~fwt(void);

	void waveletTransform(cv::Mat &input, vector<cv::Mat> &output);
	void waveletInverseTransform(vector<cv::Mat> &input, cv::Mat &output, bool lowPass = false, bool highPass = false);

private:
	fwt::Type m_type;
	int K;
	int m_width,
		m_height;
};

