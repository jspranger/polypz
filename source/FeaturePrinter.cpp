#include "..\include\FeaturePrinter.h"

FeaturePrinter::FeaturePrinter(string file)
{
	m_ostream.open(file);
	m_firstWrite = true;
}

FeaturePrinter::~FeaturePrinter(void)
{
	if (m_ostream.is_open())
		m_ostream.close();
}

bool FeaturePrinter::initRelation(string relation)
{
	if (relation.size() && m_firstWrite)
		m_relation = relation;
	return (m_ostream.good() && m_relation.size());
}

int FeaturePrinter::initAttribute(string name, string type)
{
	int result = -1;

	if (m_ostream.good() && m_relation.size() && name.size() && type.size() && m_firstWrite)
	{
		m_attrib.push_back(name);
		m_types.push_back(type);

		result = m_attrib.size() - 1;
	}

	return result;
}

bool FeaturePrinter::addData(int attribIndex, int value)
{
	bool result = false;

	if ((attribIndex >= 0) && (attribIndex < m_attrib.size()))
	{
		if (!m_data.size())
		{
			for (int i = 0; i < m_attrib.size(); i++)
				m_data.push_back("");
		}
		if (m_data[attribIndex] == "")
		{
			m_data[attribIndex] = toString(value);
			result = true;
		}
	}

	return result;
}

bool FeaturePrinter::addData(int attribIndex, double value)
{
	bool result = false;

	if ((attribIndex >= 0) && (attribIndex < m_attrib.size()))
	{
		if (!m_data.size())
		{
			for (int i = 0; i < m_attrib.size(); i++)
				m_data.push_back("");
		}
		if (m_data[attribIndex] == "")
		{
			m_data[attribIndex] = toString(value);
			result = true;
		}
	}

	return result;
}

bool FeaturePrinter::printData()
{
	bool result = false;

	if (m_attrib.size() && m_types.size() && m_data.size())
	{
		// Verify if ready to write data
		bool ready = true;
		for (int d = 0; d < m_data.size(); d++)
			if (m_data[d] == "")
				ready = false;

		if (ready)
		{
			if (m_firstWrite)
			{
				std::string m_header;

				m_header.append("@RELATION "); m_header.append(m_relation); m_header.append("\n\n");
				for (int i = 0; i < m_attrib.size(); i++)
				{
					m_header.append("@ATTRIBUTE ");
					m_header.append(m_attrib.at(i));
					m_header.append(" ");
					m_header.append(m_types.at(i));
					m_header.append("\n");
				}
				m_header.append("\n");

				m_header.append("@DATA\n");

				m_ostream.write(m_header.c_str(), m_header.size());

				m_firstWrite = false;
			}

			int n = 0;
			std::string m_fileData;

			while (n < m_data.size())
			{
				m_fileData.append(m_data[n]);
				m_data[n] = "";
				n++;
				if (n < m_data.size())
					m_fileData.append(",");
			}
			m_fileData.append("\n");
			result = true;

			m_ostream.write(m_fileData.c_str(), m_fileData.size());
			m_ostream.flush();
		}
	}

	return result;
}

string FeaturePrinter::toString(double number)
{
	char m_ch[100];
	sprintf(m_ch, "%.5f\0", number);
	string m_string(m_ch);
    return m_string;   
}
string FeaturePrinter::toString(float number)
{
	char m_ch[100];
	sprintf(m_ch, "%.5f\0", number);
	string m_string(m_ch);
    return m_string;    
}
string FeaturePrinter::toString(int number)
{
	char m_ch[100];
	sprintf(m_ch, "%d\0", number);
	string m_string(m_ch);
    return m_string;     
}