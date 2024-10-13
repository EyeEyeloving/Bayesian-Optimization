#include "GaussianProcess.h"

GaussianProcess::GaussianProcess()
	: kernel_scale_gk(1.0), kernel_sigma_gk(0.5) {

}

Eigen::RowVectorXd GaussianProcess::fitGaussianKernel(const Eigen::MatrixXd& X1, const Eigen::VectorXd& X2) {
	Eigen::RowVectorXd norm2_dist = (X1.colwise() - X2).array().square().colwise().sum();
	Eigen::RowVectorXd exponent = -0.5 * norm2_dist.array() / (kernel_sigma_gk * kernel_sigma_gk);
	return (kernel_scale_gk * kernel_scale_gk) * exponent.array().exp();
}

Eigen::MatrixXd GaussianProcess::fitGaussianKernel(const Eigen::MatrixXd& X1, const Eigen::MatrixXd& X2) {
	Eigen::MatrixXd norm2_dist_matrix = X1.colwise().squaredNorm() + X2.colwise().squaredNorm() + X1.transpose() * X2;
	return (kernel_scale_gk * kernel_scale_gk) * (-0.5 / (kernel_sigma_gk * kernel_sigma_gk) * norm2_dist_matrix.array()).exp();
}


void GaussianProcess::fit(const Eigen::MatrixXd& X_train, const Eigen::MatrixXd& Y_train) {
	predictor = X_train;
	response = Y_train;
	Eigen::MatrixXd kernel_value = fitGaussianKernel(X_train, X_train);
}