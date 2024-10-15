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

std::queue<Eigen::VectorXd*> BayesianOptimization::processInitializationData(const Eigen::VectorXd& x_lower, const Eigen::VectorXd& x_upper, const int& number_init) {
	/**在范围内进行随机选取predictor
	* matlab中randomXFeasiblePoints和initialXFeasiblePoints函数还分析了生成的随机输入是否符合既定约束
	*/

	std::queue<Eigen::VectorXd*> myQueue;
	std::mt19937 gen(42);

	/*在initialXFeasiblePoints函数中还讨论了生成的最佳的初始随机采样点*/
	for (int i = 0; i < number_init; i++) {
		Eigen::VectorXd random_predictor(x_lower.rows()); // 要用new吗？
		for (int nd = 0; nd < x_lower.rows(); nd++) {
			std::uniform_real_distribution<> dis(x_lower(nd), x_upper(nd));
			random_predictor(nd) = dis(gen);
		}
		myQueue.emplace(&random_predictor);
	}

	/*遍历队列，查看初始点生成有没有问题*/
	for (int i = 0; i < number_init; i++) {
		Eigen::VectorXd* x = myQueue.front();
		myQueue.pop();
		std::cout << x->transpose() << std::endl;
		myQueue.emplace(x);
	}

	return myQueue;
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

	/**初始化需要事先准备的观测样本集
	* 随机设置一些predictor，然后获得response，构成样本{predictor，response}
	* 样本与预测的区别在于，样本是对真实世界的采样（即执行一次真实的trial并得到结果），而预测只是通过模型基于输入所做的预测
	* 事先准备的/需要进行采样的predictor通过队列存储
	*/
	Eigen::VectorXd x_lower = data_block.colwise().minCoeff(); // 待定这样写
	Eigen::VectorXd x_upper = data_block.colwise().maxCoeff();
	int number_init = 5;
	init_try_point_queue = processInitializationData(x_lower, x_upper, number_init);

	// Eigen::VectorXd candiate_predictor(predictor_dimension);
	// candiate_predictor.setZero();

	/*初始化代理模型*/
	fitSurrogateModel();

	/**搜索最优候选点，对应matlab函数findIncumbent
	* 找到使代理模型（例如，GP）最优（例如，最大均值位置）的predictor位置及其预测response
	*/
	auto candiate = findIncumbent();

	//Eigen::VectorXd candiate_predictor;
	//Eigen::VectorXd candiate_response;

	/*基于采样函数选择下一个predictor*/
	Eigen::VectorXd candiate_predictor = findNextInAcquisitionFcn();
	
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

		/*搜索下一组最优候选点*/
		candiate = findIncumbent();

		/*基于采样函数选择下一个predictor*/
		candiate_predictor = findNextInAcquisitionFcn();

		/*贝叶斯优化步数+1*/
		++number_bayesopt;

		/*设置退出条件（待定）*/
		if (number_bayesopt == number_max_iter) optimization_finished = true;
	}
}

void BayesianOptimization::fitSurrogateModel() {
	SurrogateModel::fit(predictor_set, response_set, surrogate_model);
}

Eigen::VectorXd BayesianOptimization::fitAcquisitionFcn() {
	AcquisitionFcn::fit(acquisition_func);
	return AcquisitionFcn::getNextCandiatePoint();
}

PointIncumbent BayesianOptimization::findIncumbent() {
	Eigen::VectorXd candiate_predictor(predictor_dimension);
	Eigen::VectorXd candiate_response(response_dimension);

	// matlab使用fminbndGlobal寻找全局最优解

}

Eigen::VectorXd BayesianOptimization::findNextInAcquisitionFcn() {

	/*如果初始设置的探索点仍未完成*/
	if (!init_try_point_queue.empty()) {
		Eigen::VectorXd* candiate_predictor = init_try_point_queue.front();
		init_try_point_queue.pop();
		return *candiate_predictor;
	}

	/*如果初始探索点完成*/
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

