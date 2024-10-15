/*采样函数
设计一个tradeoff，包含利用现有规则exploitation和探索未知区域exploration；
采样函数可以认为是一种utility/loss function，即找到能够最大化这个采样函数的样本？
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
	每一个新采样点会有多大的概率能够比当前已观测到的最佳的采样点更好
	这个“更好”的衡量是什么？
	弊端在于更关注于Exploitation，所以实际会加参数来更Exploration*/
	AFcnPI fitProbabilityOfImprovement(const double FBest, const double margin);

	/* EI Function
	*/
	void fitExpectedImprovement();
};

