#ifndef __DART_HELPER_H__
#define __DART_HELPER_H__
#include "dart/dart.hpp"
#include "MySkeletonPtr.h"
#include "MyIsometry3d.h"

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Core>


namespace Eigen {

using Vector1d = Matrix<double, 1, 1>;
using Matrix1d = Matrix<double, 1, 1>;
}

std::vector<double> split_to_double(const std::string& input, int num);
Eigen::Vector1d string_to_vector1d(const std::string& input);
Eigen::Vector3d string_to_vector3d(const std::string& input);
Eigen::Vector4d string_to_vector4d(const std::string& input);
Eigen::VectorXd string_to_vectorXd(const std::string& input, int n);
Eigen::Matrix3d string_to_matrix3d(const std::string& input);

dart::dynamics::ShapePtr MakeSphereShape(double radius);
dart::dynamics::ShapePtr MakeBoxShape(const Eigen::Vector3d& size);
dart::dynamics::ShapePtr MakeCapsuleShape(double radius, double height);

dart::dynamics::Inertia MakeInertia(const dart::dynamics::ShapePtr& shape,double mass);

dart::dynamics::FreeJoint::Properties* MakeFreeJointProperties(const std::string& name,const Eigen::Isometry3d& parent_to_joint = Eigen::Isometry3d::Identity(),const Eigen::Isometry3d& child_to_joint = Eigen::Isometry3d::Identity());
dart::dynamics::PlanarJoint::Properties* MakePlanarJointProperties(const std::string& name,const Eigen::Isometry3d& parent_to_joint = Eigen::Isometry3d::Identity(),const Eigen::Isometry3d& child_to_joint = Eigen::Isometry3d::Identity());
dart::dynamics::BallJoint::Properties* MakeBallJointProperties(const std::string& name,const Eigen::Isometry3d& parent_to_joint = Eigen::Isometry3d::Identity(),const Eigen::Isometry3d& child_to_joint = Eigen::Isometry3d::Identity(),const Eigen::Vector3d& lower = Eigen::Vector3d::Constant(-2.0),const Eigen::Vector3d& upper = Eigen::Vector3d::Constant(2.0));
dart::dynamics::RevoluteJoint::Properties* MakeRevoluteJointProperties(const std::string& name,const Eigen::Vector3d& axis,const Eigen::Isometry3d& parent_to_joint = Eigen::Isometry3d::Identity(),const Eigen::Isometry3d& child_to_joint = Eigen::Isometry3d::Identity(),const Eigen::Vector1d& lower = Eigen::Vector1d::Constant(-2.0),const Eigen::Vector1d& upper = Eigen::Vector1d::Constant(2.0));
dart::dynamics::WeldJoint::Properties* MakeWeldJointProperties(const std::string& name,const Eigen::Isometry3d& parent_to_joint = Eigen::Isometry3d::Identity(),const Eigen::Isometry3d& child_to_joint = Eigen::Isometry3d::Identity());

dart::dynamics::BodyNode* MakeBodyNode(const dart::dynamics::SkeletonPtr& skeleton,dart::dynamics::BodyNode* parent,dart::dynamics::Joint::Properties* joint_properties,const std::string& joint_type,dart::dynamics::Inertia inertia);
MySkeletonPtr BuildFromFile(const std::string& path,bool create_obj=false);


MyIsometry3d FreeJoint_convertToTransform(const Eigen::Vector6d & 	_positions)
{
    return MyIsometry3d(dart::dynamics::FreeJoint::convertToTransform(_positions));
}	

Eigen::Vector6d FreeJoint_convertToPositions(Eigen::Matrix4d & 	_tf	)	
{
    Eigen::Isometry3d m;
    m.linear() = _tf.block<3, 3>(0, 0);
    m.translation() = _tf.block<3, 1>(0, 3);
    return  dart::dynamics::FreeJoint::convertToPositions(m);
}


#endif
