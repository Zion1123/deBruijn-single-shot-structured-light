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
	// ����ڲ�
	kc1 = kc;

	//ͶӰ���ڲ�
	kp2 = kp;
	cv::Mat tmp;
	hconcat(cv::Mat::eye(3, 3, CV_32FC1),
		cv::Mat::zeros(cv::Size(1, 3), CV_32FC1), tmp);

	// HC ���� 
	hc1 = kc1 * tmp;

	// ��r12��t12.t()ƴ��һ��ŵ�tmp
	hconcat(r12, t12.t(), tmp);
	// Hp����
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
