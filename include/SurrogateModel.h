/**代理模型：
* 提供对未知区域的采样点的预测；它应该足够灵活以能通过微调来模拟真实函数
* （1）关键一：代理模型必须是一个概率模型
* （2）关键二：返回一个均值预测mean prediction和对当前预测的不确定性variance prediction
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

//private: // 类的继承无法使用private
	void assumeBayesianPrior();

	void fitGaussianProcess(const Eigen::MatrixXd& predictor_set, const Eigen::MatrixXd& response_set);
};