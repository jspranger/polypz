#include "..\include\fwt.h"

fwt::fwt(fwt::Type type, int levels)
{
	m_type = type;
	K = levels;
	m_width = 0;
	m_height = 0;
}

fwt::~fwt(void) {}

void fwt::waveletTransform(cv::Mat &input, vector<cv::Mat> &output)
{
	m_width = input.cols;
	m_height = input.rows;

	output = vector<cv::Mat>();
	output.push_back(cv::Mat(m_height, m_width, CV_32FC1));
	input.convertTo(output[0], CV_32FC1);

    for (int k = 0; k < K; k++) 
    {
		output.push_back(cv::Mat(m_height >> (k + 1), m_width >> (k + 1), CV_32FC1));
		output.push_back(cv::Mat(m_height >> (k + 1), m_width >> (k + 1), CV_32FC1));
		output.push_back(cv::Mat(m_height >> (k + 1), m_width >> (k + 1), CV_32FC1));

		for (int y = 0; y < (m_height >> (k + 1)); y++)
			for (int x = 0; x < (m_width >> (k + 1)); x++)
            {
				switch (m_type)
				{
					case fwt::Haar:
					default:
						// b1
						output[(k * 3) + 1].at<float>(y, x) = (output[0].at<float>(2 * y, 2 * x) +
															   output[0].at<float>(2 * y + 1, 2 * x) -
															   output[0].at<float>(2 * y, 2 * x + 1) -
															   output[0].at<float>(2 * y + 1, 2 * x + 1))
															   * 0.5;
						// b2
						output[(k * 3) + 2].at<float>(y, x) = (output[0].at<float>(2 * y, 2 * x) -
															   output[0].at<float>(2 * y, 2 * x + 1) -
															   output[0].at<float>(2 * y + 1, 2 * x) +
															   output[0].at<float>(2 * y  + 1, 2 * x + 1))
															   * 0.5;
						// b3
						output[(k * 3) + 3].at<float>(y, x) = (output[0].at<float>(2 * y, 2 * x) +
															   output[0].at<float>(2 * y, 2 * x + 1) -
															   output[0].at<float>(2 * y + 1, 2 * x) -
															   output[0].at<float>(2 * y + 1, 2 * x + 1))
															   * 0.5;
						// bk
						output[0].at<float>(y, x) = (output[0].at<float>(2 * y, 2 * x) +
													 output[0].at<float>(2 * y, 2 * x + 1) +
													 output[0].at<float>(2 * y + 1, 2 * x) +
													 output[0].at<float>(2 * y + 1, 2 * x + 1))
													 * 0.5;
						break;
				}
            }
		output[0] = output[0](cv::Rect(0, 0, m_width >> (k + 1),m_height >> (k + 1)));
    }
}

void fwt::waveletInverseTransform(vector<cv::Mat> &input, cv::Mat &output, bool lowPass, bool highPass)
{
	float c = 0,
		  dh = 0,
		  dd = 0,
		  dv = 0;

	output = cv::Mat::zeros(m_width, m_height, CV_8UC1);
	cv::Mat temp_input,
			temp_output = cv::Mat::zeros(m_height, m_width, CV_32FC1);
	input[0].copyTo(temp_input);

	for (int k = (K - 1); k >= 0; k--)
	{
        for (int y = 0; y < (m_height >> (k + 1)); y++)
            for (int x = 0; x < (m_width >> (k + 1)); x++)
			{
				switch (m_type)
				{
					case fwt::Haar:
					default:
						if (!highPass)
							c = temp_input.at<float>(y, x); // bk
						if (!lowPass)
						{
							dh = input[(k * 3) + 1].at<float>(y, x); // b1
							dd = input[(k * 3) + 2].at<float>(y, x); // b2
							dv = input[(k * 3) + 3].at<float>(y, x); // b3
						}

						temp_output.at<float>(2 * y, 2 * x) = 0.5 * (c + dh + dd + dv);
						temp_output.at<float>(2 * y, 2 * x + 1) = 0.5 * (c - dh - dd + dv);
						temp_output.at<float>(2 * y + 1, 2 * x) = 0.5 * (c + dh - dd - dv);
						temp_output.at<float>(2 * y + 1, 2 * x + 1) = 0.5 * (c - dh + dd - dv);
						break;
				}
            }
		temp_input = cv::Mat(m_height >> k, m_width >> k, CV_32FC1);
		temp_output(cv::Rect(0, 0, m_width >> k, m_height >> k)).copyTo(temp_input);
	}
	cv::normalize(temp_output, temp_output, 0, 255, CV_MINMAX);
	temp_output.convertTo(output, CV_8UC1);
}
