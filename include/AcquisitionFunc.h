/*��������
���һ��tradeoff�������������й���exploitation��̽��δ֪����exploration��
��������������Ϊ��һ��utility/loss function�����ҵ��ܹ�����������������������
*/

#pragma once

#include <iostream>
#include <string>
#include <Eigen/Dense>

struct AFcnPI {
	Eigen::VectorXd PI;
	Eigen::VectorXd FSD;
	Eigen::VectorXd GammaX;
};

class AcquisitionFunc
{
public:
	std::string acquisition_func;

public:
	AcquisitionFunc();

	AcquisitionFunc(std::string afcn_name);

	void fit();

private:
	/* PI Function
	ÿһ���²�������ж��ĸ����ܹ��ȵ�ǰ�ѹ۲⵽����ѵĲ��������
	��������á��ĺ�����ʲô��
	�׶����ڸ���ע��Exploitation������ʵ�ʻ�Ӳ�������Exploration*/
	AFcnPI fitProbabilityOfImprovement(const double FBest, const double margin);

	/* EI Function
	*/
	void fitExpectedImprovement();
};

