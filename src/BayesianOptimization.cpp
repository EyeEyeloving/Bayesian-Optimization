#include "BayesianOptimization.h"
#include <random>

BayesianOptimization::BayesianOptimization()
	: surrogate_model("GaussianProcess"), acquisition_func("ExpectedImprovement") {

}

BayesianOptimization::BayesianOptimization(std::string surrogate, std::string acquisition)
	: surrogate_model(surrogate), acquisition_func(acquisition) {

}

Eigen::MatrixXd BayesianOptimization::validateDataInput(Eigen::MatrixXd& data_block, int& data_dimension) {
	predictor_dimension = data_dimension;
	if (data_block.rows() == predictor_dimension) return data_block;
	return data_block.transpose();
}

/*matlab中run函数中的findIncumbent函数，可能并不需要在现阶段完成解读*/
//NextPointCandiate BayesianOptimization::findNextPointCandiate(const Eigen::VectorXd& x_lower, const Eigen::VectorXd& x_upper) {
//	Eigen::VectorXd candiate_predictor(predictor_dimension);
//	Eigen::VectorXd candiate_response(response_dimension);
//
//	/*为了找到更合适的下一个采样点，matlab使用了函数fminbndGlobal和fminsearch*/ 
//
//	// 生成给定范围内的随机数
//	// int number_candiate = 1;
//	std::mt19937 gen(42);
//	for (int nd = 0; nd < predictor_dimension; nd++) {
//		std::uniform_real_distribution<> dis(x_lower(nd), x_upper(nd));
//		candiate_predictor(nd) = dis(gen);
//	}
//	// 计算采样点的函数值
//	candiate_response.setConstant(1);
//	
//	return {candiate_predictor, candiate_response};
//}

void BayesianOptimization::fit(Eigen::MatrixXd& data_block_raw, int& data_dimension) {
	/*检查输入数据*/
	Eigen::MatrixXd data_block = validateDataInput(data_block_raw, data_dimension);

}