#pragma once

#include <vector>
#include <fstream>
#include <sstream>

using namespace std;

class FeaturePrinter
{
public:
	FeaturePrinter(string file = "file.arff");
	~FeaturePrinter(void);

	bool initRelation(string relation = "standard");
	int initAttribute(string name, string type);

	bool addData(int attribIndex, int value);
	bool addData(int attribIndex, double value);

	bool printData(void);

private:
	ofstream m_ostream;
	bool m_firstWrite;

	string m_relation;
	vector<string> m_data;
	vector<string> m_attrib,
				   m_types,
				   m_dataRow;

	string toString(double number);
	string toString(float number);
	string toString(int number);
};

