#include "..\include\FeatureExtractor.h"

FeatureExtractor::FeatureExtractor(string imagePath,
								   int waveletLevels,
								   vector<itk::Offset<2> > offsets)
{
	m_status = true;

	K = waveletLevels;
	m_offsets = offsets;

	m_fwt = new fwt(fwt::Haar, K);

	cv::Mat I = cv::imread(imagePath, CV_LOAD_IMAGE_COLOR);
	if ((I.data) && (I.channels() == 3))
	{
#ifdef HSV
		cv::cvtColor(I,I,CV_BGR2HSV);
#endif
		// Split channels
		cv::Mat Ci[3];
		cv::split(I, Ci);

		// Process wavelet for each channel, gather co-ocurrence matrix and metrics
		for (int c = Color::blue; c <= Color::red; c++)
		{
			m_selection.push_back(vector<Features>());
			
#ifdef HSV
			
			cv::cvtColor(I,I,CV_BGR2HSV);
			if (c == 0)
				cv::normalize(Ci[c], Ci[c], 0, 179, CV_MINMAX);
			else cv::normalize(Ci[c], Ci[c], 0, 255, CV_MINMAX);
#else
			cv::normalize(Ci[c], Ci[c], 0, 255, CV_MINMAX);
#endif

			// Wavelet ______________________________________________________________________________________
			int m_max_w = cv::max(Ci[(Color)c].cols, Ci[(Color)c].rows),
				m_power_w = 0;
			while (((int)pow(2.0, (double)m_power_w)) < m_max_w)
				m_power_w++;

			int m_total_width = ((int)pow(2.0, (double)m_power_w)),
				m_horizontal_margin = (int)((m_total_width - Ci[(Color)c].rows) / 2.0),
				m_vertical_margin = (int)((m_total_width - Ci[(Color)c].cols) / 2.0);

			cv::Mat m_input;
			vector<cv::Mat> m_output;
			cv::copyMakeBorder(Ci[(Color)c],
							   m_input,
							   m_vertical_margin,
							   m_vertical_margin,
							   m_horizontal_margin,
							   m_horizontal_margin,
							   cv::BORDER_CONSTANT,
							   cv::Scalar(0, 0, 0));
			m_fwt->waveletTransform(m_input, m_output);

			cv::Mat m_detail;
			m_fwt->waveletInverseTransform(m_output, m_detail, false, true);
			cv::Rect m_roi(m_horizontal_margin,
						   m_vertical_margin,
						   Ci[(Color)c].cols,
						   Ci[(Color)c].rows);
			cv::Mat m_waveletReconstruction;
			m_detail(m_roi).copyTo(m_waveletReconstruction);
			// ______________________________________________________________________________________________

			if (m_waveletReconstruction.size)
			{
				itkImage::Pointer m_image = itk::OpenCVImageBridge::CVMatToITKImage<itkImage>(m_waveletReconstruction);
				for (int o = 0; o < m_offsets.size(); o++)
				{
					itkCoocurrenceMatrixCalculator::Pointer m_coocurrenceCalculator = itkCoocurrenceMatrixCalculator::New();
					itkTextureFeatureCalculator::Pointer m_featureCalculator = itkTextureFeatureCalculator::New();

#ifdef HSV
					if (c == 0)
						m_coocurrenceCalculator->SetPixelValueMinMax(0, 179);
					else m_coocurrenceCalculator->SetPixelValueMinMax(0, 255);
#else
					m_coocurrenceCalculator->SetPixelValueMinMax(0, 255);
#endif
					m_coocurrenceCalculator->SetInput(m_image);
					m_coocurrenceCalculator->SetOffset(m_offsets.at(o));
					m_coocurrenceCalculator->Update();

					m_featureCalculator->SetInput(m_coocurrenceCalculator->GetOutput());
					m_featureCalculator->Update();

					// DEBUG _________________________________________________________________________
					//itkHistogramDebugger::Pointer m_debug = itkHistogramDebugger::New();
					//m_debug->SetInput(m_coocurrenceCalculator->GetOutput());
					//m_debug->Update();

					//cv::Mat mimg = itk::OpenCVImageBridge::ITKImageToCVMat(m_debug->GetOutput());
					//imshow("CO-OCURRENCE_MATRIX", mimg);
					//cv::waitKey();
					// _______________________________________________________________________________

					Features f;
					f.Energy = m_featureCalculator->GetFeature(itkTextureFeatureCalculator::Energy);
					f.Entropy = m_featureCalculator->GetFeature(itkTextureFeatureCalculator::Entropy);
					f.Correlation = m_featureCalculator->GetFeature(itkTextureFeatureCalculator::Correlation);
					f.DifferenceMoment = m_featureCalculator->GetFeature(itkTextureFeatureCalculator::InverseDifferenceMoment);
					f.Inertia = m_featureCalculator->GetFeature(itkTextureFeatureCalculator::Inertia);
					f.ClusterShade = m_featureCalculator->GetFeature(itkTextureFeatureCalculator::ClusterShade);
					f.ClusterProminence = m_featureCalculator->GetFeature(itkTextureFeatureCalculator::ClusterProminence);
					f.HaralickCorrelation = m_featureCalculator->GetFeature(itkTextureFeatureCalculator::HaralickCorrelation);

					cout << f.Energy << "|" << f.Entropy << "|" << f.Correlation << "|" <<
							f.DifferenceMoment << "|" << f.Inertia << "|" << f.ClusterShade << "|" <<
							f.ClusterProminence << "|" << f.HaralickCorrelation << endl;

					m_selection.at(c).push_back(f);
				}
			} else m_status = false;
		}
	} else m_status = false;
}

FeatureExtractor::~FeatureExtractor(void)
{
	delete m_fwt;
}

//void  FeatureExtractor::adaptiveMedianBlur(cv::InputArray input, cv::OutputArray output, int ksize, int smax)
//{
//	int w = input.getMat().cols,
//		h = input.getMat().rows;
//
//	output.create(h, w, CV_8UC1);
//
//	for (int i = 0; i < w; i++)
//		for (int j = 0; j < h; j++)
//		{
//			int sxy = ksize;
//			bool proceed = false;
//
//			while ((sxy <= smax) && (!proceed))
//			{
//				std::vector<unsigned char> m_list;
//				unsigned char zxy = input.getMat().at<unsigned char>(i, j),
//					  zmin = numeric_limits<unsigned char>::max(),
//					  zmax = numeric_limits<unsigned char>::min(),
//					  zmed = zxy;
//
//				for (int x = 0; x < sxy; x++)
//					for (int y = 0; y < sxy; y++)
//						if (((x - 1 + i) >= 0) && ((y - 1 + j) >= 0) &&
//							((x - 1 + i) < w) & ((y - 1 + j) < h))
//						{
//							unsigned char m_value = input.getMat().at<unsigned char>((x - 1 + i), (y - 1 + j));
//							m_list.push_back(m_value);
//							if (m_value > zmax) zmax = m_value;
//							if (m_value < zmin) zmin = m_value;
//						}
//				std::sort(m_list.begin(), m_list.end());
//
//				if (m_list.size())
//				{
//					// Sorted list count is even?
//					if (!(m_list.size() % 2))
//						zmed = (unsigned char)((m_list[(int)trunc((m_list.size() - 1) / 2.0)] +
//							   m_list[(int)round((m_list.size() - 1) / 2.0)]) / 2.0);
//					// Sorted list count is odd?
//					else
//						zmed = (unsigned char)m_list[(int)(m_list.size() / 2.0)];
//				}
//
//				unsigned char A1 = zmed - zmin,
//							  A2 = zmed - zmax;
//				if ((A1 > 0.0) && (A2 < 0.0))
//				{
//					unsigned char B1 = zxy - zmin,
//								  B2 = zxy - zmax;
//					if ((B1 > 0.0) && (B2 < 0.0))
//						output.getMatRef().at<unsigned char>(i, j) = zxy;
//					else
//						output.getMatRef().at<unsigned char>(i, j) = zmed;
//					proceed = true;
//				}
//				else
//				{
//					sxy += 2;
//					if (sxy > smax)
//					{
//						output.getMatRef().at<unsigned char>(i, j) = zmed;
//						proceed = true;
//					}
//				}
//			}
//		}
//		cv::normalize(output.getMat(), output.getMatRef(), 0, 255, CV_MINMAX);
//}

Features FeatureExtractor::F(Color c, int a)
{
	return m_selection.at(c).at(a);
}

FeatureMV FeatureExtractor::FMV(void)
{
	FeatureMV m_featureMV = { 0, 0,
							  0, 0,
							  0, 0,
							  
							  0, 0,
							  0, 0,
							  0, 0,

							  0, 0,
							  0, 0,
							  0, 0,

							  0, 0,
							  0, 0,
							  0, 0,

							  0, 0,
							  0, 0,
							  0, 0,

							  0, 0,
							  0, 0,
							  0, 0,

							  0, 0,
							  0, 0,
							  0, 0,

							  0, 0,
							  0, 0,
							  0, 0
							};

	if (m_offsets.size())
	{
		// Compute means
		for (int a = 0; a < m_offsets.size(); a++)
		{
			Features m_features_c1 = F((Color)0, a),
					 m_features_c2 = F((Color)1, a),
					 m_features_c3 = F((Color)2, a);

			m_featureMV.C1_EnergyMean += m_features_c1.Energy;
			m_featureMV.C2_EnergyMean += m_features_c2.Energy;
			m_featureMV.C3_EnergyMean += m_features_c3.Energy;

			m_featureMV.C1_EntropyMean += m_features_c1.Entropy;
			m_featureMV.C2_EntropyMean += m_features_c2.Entropy;
			m_featureMV.C3_EntropyMean += m_features_c3.Entropy;

			m_featureMV.C1_CorrelationMean += m_features_c1.Correlation;
			m_featureMV.C2_CorrelationMean += m_features_c2.Correlation;
			m_featureMV.C3_CorrelationMean += m_features_c3.Correlation;

			m_featureMV.C1_DifferenceMomentMean += m_features_c1.DifferenceMoment;
			m_featureMV.C2_DifferenceMomentMean += m_features_c2.DifferenceMoment;
			m_featureMV.C3_DifferenceMomentMean += m_features_c3.DifferenceMoment;

			m_featureMV.C1_InertiaMean += m_features_c1.Inertia;
			m_featureMV.C2_InertiaMean += m_features_c2.Inertia;
			m_featureMV.C3_InertiaMean += m_features_c3.Inertia;

			m_featureMV.C1_ClusterShadeMean += m_features_c1.ClusterShade;
			m_featureMV.C2_ClusterShadeMean += m_features_c2.ClusterShade;
			m_featureMV.C3_ClusterShadeMean += m_features_c3.ClusterShade;

			m_featureMV.C1_ClusterProminenceMean += m_features_c1.ClusterProminence;
			m_featureMV.C2_ClusterProminenceMean += m_features_c2.ClusterProminence;
			m_featureMV.C3_ClusterProminenceMean += m_features_c3.ClusterProminence;

			m_featureMV.C1_HaralickCorrelationMean += m_features_c1.HaralickCorrelation;
			m_featureMV.C2_HaralickCorrelationMean += m_features_c2.HaralickCorrelation;
			m_featureMV.C3_HaralickCorrelationMean += m_features_c3.HaralickCorrelation;
		}

		m_featureMV.C1_EnergyMean /= m_offsets.size();
		m_featureMV.C2_EnergyMean /= m_offsets.size();
		m_featureMV.C3_EnergyMean /= m_offsets.size();

		m_featureMV.C1_EntropyMean /= m_offsets.size();
		m_featureMV.C2_EntropyMean /= m_offsets.size();
		m_featureMV.C3_EntropyMean /= m_offsets.size();

		m_featureMV.C1_CorrelationMean /= m_offsets.size();
		m_featureMV.C2_CorrelationMean /= m_offsets.size();
		m_featureMV.C3_CorrelationMean /= m_offsets.size();

		m_featureMV.C1_DifferenceMomentMean /= m_offsets.size();
		m_featureMV.C2_DifferenceMomentMean /= m_offsets.size();
		m_featureMV.C3_DifferenceMomentMean /= m_offsets.size();

		m_featureMV.C1_InertiaMean /= m_offsets.size();
		m_featureMV.C2_InertiaMean /= m_offsets.size();
		m_featureMV.C3_InertiaMean /= m_offsets.size();

		m_featureMV.C1_ClusterShadeMean /= m_offsets.size();
		m_featureMV.C2_ClusterShadeMean /= m_offsets.size();
		m_featureMV.C3_ClusterShadeMean /= m_offsets.size();

		m_featureMV.C1_ClusterProminenceMean /= m_offsets.size();
		m_featureMV.C2_ClusterProminenceMean /= m_offsets.size();
		m_featureMV.C3_ClusterProminenceMean /= m_offsets.size();

		m_featureMV.C1_HaralickCorrelationMean /= m_offsets.size();
		m_featureMV.C2_HaralickCorrelationMean /= m_offsets.size();
		m_featureMV.C3_HaralickCorrelationMean /= m_offsets.size();

		// Compute variances
		for (int a = 0; a < m_offsets.size(); a++)
		{
			Features m_features_c1 = F((Color)0, a),
					 m_features_c2 = F((Color)1, a),
					 m_features_c3 = F((Color)2, a);

			m_featureMV.C1_EnergyVariance += pow(m_features_c1.Energy - m_featureMV.C1_EnergyMean, 2.0);
			m_featureMV.C2_EnergyVariance += pow(m_features_c2.Energy - m_featureMV.C2_EnergyMean, 2.0);
			m_featureMV.C3_EnergyVariance += pow(m_features_c3.Energy - m_featureMV.C3_EnergyMean, 2.0);

			m_featureMV.C1_EntropyVariance += pow(m_features_c1.Entropy - m_featureMV.C1_EntropyMean, 2.0);
			m_featureMV.C2_EntropyVariance += pow(m_features_c2.Entropy - m_featureMV.C2_EntropyMean, 2.0);
			m_featureMV.C3_EntropyVariance += pow(m_features_c3.Entropy - m_featureMV.C3_EntropyMean, 2.0);

			m_featureMV.C1_CorrelationVariance += pow(m_features_c1.Correlation - m_featureMV.C1_CorrelationMean, 2.0);
			m_featureMV.C2_CorrelationVariance += pow(m_features_c2.Correlation - m_featureMV.C2_CorrelationMean, 2.0);
			m_featureMV.C3_CorrelationVariance += pow(m_features_c3.Correlation - m_featureMV.C3_CorrelationMean, 2.0);

			m_featureMV.C1_DifferenceMomentVariance += pow(m_features_c1.DifferenceMoment - m_featureMV.C1_DifferenceMomentMean, 2.0);
			m_featureMV.C2_DifferenceMomentVariance += pow(m_features_c2.DifferenceMoment - m_featureMV.C2_DifferenceMomentMean, 2.0);
			m_featureMV.C3_DifferenceMomentVariance += pow(m_features_c3.DifferenceMoment - m_featureMV.C3_DifferenceMomentMean, 2.0);

			m_featureMV.C1_InertiaVariance += pow(m_features_c1.Inertia - m_featureMV.C1_InertiaMean, 2.0);
			m_featureMV.C2_InertiaVariance += pow(m_features_c2.Inertia - m_featureMV.C2_InertiaMean, 2.0);
			m_featureMV.C3_InertiaVariance += pow(m_features_c3.Inertia - m_featureMV.C3_InertiaMean, 2.0);

			m_featureMV.C1_ClusterShadeVariance += pow(m_features_c1.ClusterShade - m_featureMV.C1_ClusterShadeMean, 2.0);
			m_featureMV.C2_ClusterShadeVariance += pow(m_features_c2.ClusterShade - m_featureMV.C2_ClusterShadeMean, 2.0);
			m_featureMV.C3_ClusterShadeVariance += pow(m_features_c3.ClusterShade - m_featureMV.C3_ClusterShadeMean, 2.0);

			m_featureMV.C1_ClusterProminenceVariance += pow(m_features_c1.ClusterProminence - m_featureMV.C1_ClusterProminenceMean, 2.0);
			m_featureMV.C2_ClusterProminenceVariance += pow(m_features_c2.ClusterProminence - m_featureMV.C2_ClusterProminenceMean, 2.0);
			m_featureMV.C3_ClusterProminenceVariance += pow(m_features_c3.ClusterProminence - m_featureMV.C3_ClusterProminenceMean, 2.0);

			m_featureMV.C1_HaralickCorrelationVariance += pow(m_features_c1.HaralickCorrelation - m_featureMV.C1_HaralickCorrelationMean, 2.0);
			m_featureMV.C2_HaralickCorrelationVariance += pow(m_features_c2.HaralickCorrelation - m_featureMV.C2_HaralickCorrelationMean, 2.0);
			m_featureMV.C3_HaralickCorrelationVariance += pow(m_features_c3.HaralickCorrelation - m_featureMV.C3_HaralickCorrelationMean, 2.0);
		}

		m_featureMV.C1_EnergyVariance /= m_offsets.size();
		m_featureMV.C2_EnergyVariance /= m_offsets.size();
		m_featureMV.C3_EnergyVariance /= m_offsets.size();

		m_featureMV.C1_EntropyVariance /= m_offsets.size();
		m_featureMV.C2_EntropyVariance /= m_offsets.size();
		m_featureMV.C3_EntropyVariance /= m_offsets.size();

		m_featureMV.C1_CorrelationVariance /= m_offsets.size();
		m_featureMV.C2_CorrelationVariance /= m_offsets.size();
		m_featureMV.C3_CorrelationVariance /= m_offsets.size();

		m_featureMV.C1_DifferenceMomentVariance /= m_offsets.size();
		m_featureMV.C2_DifferenceMomentVariance /= m_offsets.size();
		m_featureMV.C3_DifferenceMomentVariance /= m_offsets.size();

		m_featureMV.C1_InertiaVariance /= m_offsets.size();
		m_featureMV.C2_InertiaVariance /= m_offsets.size();
		m_featureMV.C3_InertiaVariance /= m_offsets.size();

		m_featureMV.C1_ClusterShadeVariance /= m_offsets.size();
		m_featureMV.C2_ClusterShadeVariance /= m_offsets.size();
		m_featureMV.C3_ClusterShadeVariance /= m_offsets.size();

		m_featureMV.C1_ClusterProminenceVariance /= m_offsets.size();
		m_featureMV.C2_ClusterProminenceVariance /= m_offsets.size();
		m_featureMV.C3_ClusterProminenceVariance /= m_offsets.size();

		m_featureMV.C1_HaralickCorrelationVariance /= m_offsets.size();
		m_featureMV.C2_HaralickCorrelationVariance /= m_offsets.size();
		m_featureMV.C3_HaralickCorrelationVariance /= m_offsets.size();
	}

	return m_featureMV;
}