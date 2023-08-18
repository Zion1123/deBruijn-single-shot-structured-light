#include "CameraArguments.h"

CameraArguments* CameraArguments::instance = nullptr;

/**
 * \brief 
 */
CameraArguments::CameraArguments()
{
	//r12 = cv::Mat::zeros(cv::Size(3, 3), CV_8UC1);
}

CameraArguments::CameraArguments(cv::Mat r, cv::Mat t, cv::Mat kc, cv::Mat kp)
{
	r12 = r;
	t12 = t;
	// 相机内参
	kc1 = kc;

	//投影仪内参
	kp2 = kp;
	cv::Mat tmp;
	hconcat(cv::Mat::eye(3, 3, CV_32FC1),
		cv::Mat::zeros(cv::Size(1, 3), CV_32FC1), tmp);

	// HC 矩阵， 
	hc1 = kc1 * tmp;

	// 将r12和t12.t()拼在一起放到tmp
	hconcat(r12, t12.t(), tmp);
	// Hp矩阵
	hp2 = kp2 * tmp;
}

CameraArguments* CameraArguments::getInstance(cv::Mat r, cv::Mat t, cv::Mat kc, cv::Mat kp)
{
	if (instance == nullptr) instance = new CameraArguments(r, t, kc, kp);
	return instance;
}

CameraArguments* CameraArguments::getInstance()
{
	if (instance) return instance;
}

CameraArguments::~CameraArguments()
= default;

cv::Mat CameraArguments::getR() const
{
	return r12;
}

cv::Mat CameraArguments::getT() const
{
	return t12;
}

cv::Mat CameraArguments::getKc() const
{
	return kc1;
}

cv::Mat CameraArguments::getKp() const
{
	return kp2;
}


cv::Mat CameraArguments::getHc() const
{
	return hc1;
}

cv::Mat CameraArguments::getHp() const
{
	return hp2;
}
