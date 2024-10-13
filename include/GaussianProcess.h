/**高斯过程
* 描述了采样点上目标函数值的后验概率分布（即给定采样点，基于该点的目标函数值的条件概率分布）
* 这表明该点的真实目标值可能由多个潜在目标函数采样得到，所以GP将该采样点的真实函数值视为一种随机变量
* 因此，GP在采样点的切面上是函数值随机变量的分布，而在采样点的取值范围内，GP可以作为是多个潜在目标函数的代理模型
* GP是一种非参数模型，并可以通过mean function和covariance/kernel function完全定义
* kernel function的选择影响GP对目标函数的模拟，包含了对目标函数的理解或先验知识
* GP最大的弊端是当数据量较大时的计算复杂度
*/

#pragma once

#include <Eigen/Dense>

class GaussianProcess
{
private:
	/*收集已知采样点的数据集的基本信息，包括预测子和响应子*/
	int predict_dimension;
	Eigen::MatrixXd predictor;
	int response_dimension;
	Eigen::MatrixXd response;

	/*Gaussian Kernel*/
	double kernel_sigma_gk;
	double kernel_scale_gk;

public:
	GaussianProcess();

	void fit(const Eigen::MatrixXd& X_train, const Eigen::MatrixXd& Y_train);

	void predict();

private:
	Eigen::RowVectorXd fitGaussianKernel(const Eigen::MatrixXd& X1, const Eigen::VectorXd& X2);
	
	Eigen::MatrixXd fitGaussianKernel(const Eigen::MatrixXd& X1, const Eigen::MatrixXd& X2);
};

