/*��������
���һ��tradeoff�������������й���exploitation��̽��δ֪����exploration��
��������������Ϊ��һ��utility/loss function�����ҵ��ܹ�����������������������
*/

#pragma once

#include <iostream>
#include <string>
#include <Eigen/Dense>

struct objectiveFcnGP {
	double beta;
	double sigma;
};

struct AFcnPI {
	Eigen::RowVectorXd PI;
	Eigen::RowVectorXd FSD;
	Eigen::RowVectorXd GammaX;
};

struct AFcnEI {
	Eigen::RowVectorXd EI;
};

class AcquisitionFcn
{
public:
	std::string acquisition_func;

private:
	Eigen::VectorXd candiate_predictor;

public:
	AcquisitionFcn();

	AcquisitionFcn(std::string afcn_name);

	void fit(std::string afcn_name);

	Eigen::VectorXd getNextCandiatePoint() const;

private:
	/* PI Function
	ÿһ���²�������ж��ĸ����ܹ��ȵ�ǰ�ѹ۲⵽����ѵĲ��������
	��������á��ĺ�����ʲô��
	�׶����ڸ���ע��Exploitation������ʵ�ʻ�Ӳ�������Exploration*/
	AFcnPI fitProbabilityOfImprovement(const Eigen::MatrixXd domainX, objectiveFcnGP obj, const double FBest, const double margin);

	/* EI Function
	*/
	AFcnEI fitExpectedImprovement(const Eigen::MatrixXd domainX, objectiveFcnGP obj, double FBest);
};

