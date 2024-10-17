#include "BayesianOptimization.h"
#include <random>
#include <NLopt/nlopt.hpp>

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

std::queue<Eigen::VectorXd*> BayesianOptimization::processInitializationData(const int& number_init) {
	/**在范围内进行随机选取predictor
	* matlab中randomXFeasiblePoints和initialXFeasiblePoints函数还分析了生成的随机输入是否符合既定约束
	*/

	std::queue<Eigen::VectorXd*> myQueue;
	std::mt19937 gen(42);

	/*在initialXFeasiblePoints函数中还讨论了生成的最佳的初始随机采样点*/
	for (int i = 0; i < number_init; i++) {
		Eigen::VectorXd random_predictor(predictor_domainMin.size()); // 要用new吗？
		for (int nd = 0; nd < predictor_domainMin.size(); nd++) {
			std::uniform_real_distribution<> dis(predictor_domainMin[nd], predictor_domainMax[nd]);
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
	init_try_point_queue = processInitializationData(number_init);

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

	/*matlab使用fminbndGlobal寻找全局最优解，避免找到局部最优*/

	return { candiate_predictor, candiate_response };
}

Eigen::VectorXd BayesianOptimization::findNextInAcquisitionFcn() {

	/*如果初始设置的探索点仍未完成*/
	if (!init_try_point_queue.empty()) {
		Eigen::VectorXd* candiate_predictor = init_try_point_queue.front();
		init_try_point_queue.pop();
		return *candiate_predictor;
	}

	/*如果初始探索点完成*/

	// matlab中shouldChooseRandomPoint函数
	// 作用是决定是否应该选择一个随机来探索目标函数空间，而不是依赖现有的代理模型进行预测和优化。
	//
	// matlab中legalizePoints函数
    // 作用是将原始输入（经过转换的点）映射回可行的解空间，并确保这些解符合问题定义的所有约束。
	//
	// matlab中ProbAllConstraintsSatisfied函数
	// 计算有约束的概率

	/**全局最优化
	* 基于采集函数，找到采样函数中使得采样函数概率最大的点，作为下一个探索点
	*/
	Eigen::VectorXd candiate_predictor = fitAcquisitionFcn();

	return candiate_predictor;
}

std::vector<PointIncumbent> BayesianOptimization::findFBestGlobal(int& num_initial_points, int& num_topK_points) {
	std::vector<PointIncumbent> candiatesN; // 存储所有的候选点
	std::vector<double> scores; // 存储候选点的目标函数值

	/*基于NLopt库进行全局优化*/
	nlopt::opt opt(nlopt::GN_CRS2_LM, predictor_domainMin.size());
	opt.set_lower_bounds(predictor_domainMin); // 设置优化边界
	opt.set_upper_bounds(predictor_domainMax);
	// 
	opt.set_max_objective([this](const Eigen::VectorXd& predictor) {
		return SurrogateModel::predict(predictor);
		}, nullptr);
	// 设置终止条件，例如容差和最大迭代次数
	opt.set_xtol_rel(1e-6);
	opt.set_maxeval(100); // 最大迭代次数

	std::mt19937 gen(42);
	// int num_initial_points = 1;
	// 对每个初始点执行优化
	for (int i = 0; i < num_initial_points; i++) {
		std::vector<double> x0(predictor_dimension);

		// 为每个维度生成一个随机初始猜测
		for (int nd = 0; nd < predictor_dimension; nd++) {
			std::uniform_real_distribution<> dis(predictor_domainMin[nd], predictor_domainMax[nd]);
			x0[nd] = dis(gen);
		}

		PointIncumbent candiate;
		double fbest;
		// 执行优化
		nlopt::result result = opt.optimize(x0, fbest);
		// double转换为VectorXd
		candiate.candiate_predictor = Eigen::Map<Eigen::VectorXd>(x0.data(), x0.size()); 
		candiate.candiate_response = SurrogateModel::predict(candiate.candiate_predictor);
		candiatesN.emplace_back(candiate);
		scores.emplace_back(fbest); // 初始值是什么？
	}

	/*选择前k个最佳点*/
	std::vector<PointIncumbent> candicatesK;
	std::vector<int> index(candiatesN.size()); // 创建candidatesN的索引
	for (int i = 0; i < candiatesN.size(); i++) {
		index[i] = i;
	}
	std::sort(index.begin(), index.end(), [&scores](int a, int b) {
		return scores[a] > scores[b]; 
		});
	for (int k = 0; k < num_topK_points; k++) {
		candicatesK.emplace_back(candiatesN[index[k]]);
	}

	return candicatesK;
}

