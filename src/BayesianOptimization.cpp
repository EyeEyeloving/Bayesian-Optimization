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

Eigen::VectorXd BayesianOptimization::callObjectiveFcn(const Eigen::VectorXd& candiate_predictor) {
	/*在现实世界中采样获得有噪声的目标函数值*/
	Eigen::VectorXd evaluation_response(response_dimension);
	return evaluation_response.setZero();
}

void BayesianOptimization::augmentObservationSet(const Eigen::VectorXd candiate_predictor, const Eigen::VectorXd candiate_response) {
	// 这里需要用拷贝，如果是引用将导致外部修改candiate_predictor值也导致set的值发生修改
	predictor_set << candiate_predictor;
	response_set << candiate_response;
}

void BayesianOptimization::updateTrace() {

}

void BayesianOptimization::fit(Eigen::MatrixXd& data_block_raw, int& data_dimension) {
	
	/*检查输入数据*/

	Eigen::MatrixXd data_block = validateDataInput(data_block_raw, data_dimension);
	
	/*初始化代理模型*/
	//Eigen::VectorXd candiate_predictor = findNextInAcquisitionFcn();
	Eigen::VectorXd candiate_predictor(predictor_dimension);
	candiate_predictor.setZero();
	
	/*进入贝叶斯优化*/

	while (!optimization_finished) {
		/*对目标函数进行采样*/
		Eigen::VectorXd evaluation_resopnse = callObjectiveFcn(candiate_predictor);
		/*增广已知数据矩阵*/
		augmentObservationSet(candiate_predictor, evaluation_resopnse);
		/*重新训练代理模型*/
		fitSurrogateModel();
		/*记录贝叶斯优化过程*/
		updateTrace();

		/*选择下一个候选点*/
		candiate_predictor = findNextInAcquisitionFcn();
		++number_bayesopt;
	}
}

void BayesianOptimization::fitSurrogateModel() {
	SurrogateModel::fit(predictor_set, response_set, surrogate_model);
}

Eigen::VectorXd BayesianOptimization::fitAcquisitionFcn() {
	AcquisitionFcn::fit(acquisition_func);
	return AcquisitionFcn::getNextCandiatePoint();
}

Eigen::VectorXd BayesianOptimization::findNextInAcquisitionFcn() {
	// 获得Incumbent的目的？

	// 如果没有候选点std::queue需要callObjectivefcn了



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

