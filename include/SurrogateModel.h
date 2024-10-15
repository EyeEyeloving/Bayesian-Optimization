/**����ģ�ͣ�
* �ṩ��δ֪����Ĳ������Ԥ�⣻��Ӧ���㹻�������ͨ��΢����ģ����ʵ����
* ��1���ؼ�һ������ģ�ͱ�����һ������ģ��
* ��2���ؼ���������һ����ֵԤ��mean prediction�ͶԵ�ǰԤ��Ĳ�ȷ����variance prediction
* 
*/

#pragma once

#include <iostream>
#include <string>
#include <Eigen/Dense>

struct GPAgent {
	Eigen::MatrixXd mu_gp;
	Eigen::MatrixXd cov_gp;
};

class SurrogateModel
{
private:
	bool model_strategy_set = false;

public:
	SurrogateModel();

	void fit(const Eigen::MatrixXd& predictor_set, const Eigen::MatrixXd& response_set, std::string& surrogate_name);

//private: // ��ļ̳��޷�ʹ��private
	void assumeBayesianPrior();

	void fitGaussianProcess(const Eigen::MatrixXd& predictor_set, const Eigen::MatrixXd& response_set);
};