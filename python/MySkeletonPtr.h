#ifndef __MySKELETONPTR_H__
#define __MySKELETONPTR_H__
#include "dart/dart.hpp"
#include "MyBodyNodePtr.h"

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Core>
#include <utility>
#include <string>

namespace py = pybind11;

using namespace dart::dynamics;
class MyBodyNodePtr;


class MySkeletonPtr: public SkeletonPtr{
public:
    MySkeletonPtr();
    MySkeletonPtr(const SkeletonPtr& ptr);
    MySkeletonPtr(std::nullptr_t ptr):SkeletonPtr(ptr){};

    const std::string& getRootBodyNodeParentJointType(){
        return (*this)->getRootBodyNode()->getParentJoint()->getType();
    }
    std::size_t getNumDofs() const{
        return (*this)->getNumDofs();
    }

    void clearConstraintImpulses()
    {
        (*this)->clearConstraintImpulses();
    }

    void clearInternalForces(){
        (*this)->clearInternalForces();
    }

    void clearExternalForces(){
        (*this)->clearExternalForces();
    }

    void setPositions(const Eigen::VectorXd& _positions){
        (*this)->setPositions(_positions);
    }

    void setVelocities(const Eigen::VectorXd& _velocities){
        (*this)->setVelocities(_velocities);
    }

    Eigen::VectorXd getPositions()
    {
        return (*this)->getPositions();
    }

    Eigen::VectorXd getVelocities()
    {
        return (*this)->getVelocities();
    }

    void computeForwardKinematics(bool _updateTransforms=true, bool _updateVels=true, bool _updateAccs=true){
        (*this)->computeForwardKinematics(_updateTransforms, _updateVels, _updateAccs);
    }


    Eigen::Isometry3d getBodyNode_i_getTransform(int i) const{
        return (*this)->getBodyNode(i)->getTransform();
    }

    Eigen::Matrix4d getBodyNode_i_getTransform_to_matrix(int i) const{
        Eigen::Isometry3d a = (*this)->getBodyNode(i)->getTransform();
        Eigen::Matrix4d b = Eigen::Matrix4d::Zero(4, 4);
        b.block<3, 3>(0, 0) = a.linear();
        b.block<3, 1>(0, 3) = a.translation();
        return b;
    }

    Eigen::Matrix4d getBodyNode_i_getTransform_inverse_to_matrix(int i) const{
        Eigen::Isometry3d a = (*this)->getBodyNode(i)->getTransform().inverse();
        Eigen::Matrix4d b = Eigen::Matrix4d::Zero(4, 4);
        b.block<3, 3>(0, 0) = a.linear();
        b.block<3, 1>(0, 3) = a.translation();
        return b;
    }

    const Eigen::Isometry3d& getBodyNode_i_getParentJoint_getTransformFromChildBodyNode(int i) const
    {
        return (*this)->getBodyNode(i)->getParentJoint()->getTransformFromChildBodyNode();
    }

    double getBodyNode0TransformTranslation_y()
    {
        return (*this)->getBodyNode(0)->getTransform().translation()[1];
    }

    Eigen::Vector3d getBodyNode_i_getCOM_root(int i){
        dart::dynamics::BodyNode* root = (*this)->getBodyNode(0);
        return (*this)->getBodyNode(i)->getCOM(root);
    }

    Eigen::Vector3d getBodyNode_0_getCOMLinearVelocity(){
        dart::dynamics::BodyNode* root = (*this)->getBodyNode(0);
        return root->getCOMLinearVelocity();
    }

    Eigen::Vector3d getBodyNode_i_getCOMLinearVelocity(int i){
        dart::dynamics::BodyNode* root = (*this)->getBodyNode(i);
        return root->getCOMLinearVelocity();
    }

    std::size_t getNumBodyNodes() const{
        return (*this)->getNumBodyNodes();
    }

    double getRootBodyNode_getCOM_y()
    {
        return (*this)->getRootBodyNode()->getCOM()[1];
    }

    void setForces(const Eigen::VectorXd &_forces)
    {
        (*this)->setForces(_forces);
    }

    Eigen::VectorXd getPositionDifferences(const Eigen::VectorXd &_q2, const Eigen::VectorXd &_q1)
    {
        return (*this)->getPositionDifferences(_q2, _q1);
    }

	std::size_t getBodyNodeByName_getParentJoint_getIndexInSkeleton(const std::string &name, int index)
    {
        return (*this)->getBodyNode(name)->getParentJoint()->getIndexInSkeleton(index);
    }

    const std::string& getBodyNodeByName_getParentJoint_getType(const std::string &name)
    {
        return (*this)->getBodyNode(name)->getParentJoint()->getType();
    }    

    Eigen::Vector3d getCOM() {
        return (*this)->getCOM();
    }

    MyBodyNodePtr getBodyNode(const std::string & 	name);

    double 	getTimeStep () const{
        return (*this)->getTimeStep();
    }

    MyBodyNodePtr getBodyNode_i(	std::size_t 	_idx	);

    Eigen::Matrix<double, 3, 1> getLinearJacobian(MyBodyNodePtr _node_ptr, 	const Eigen::Vector3d& _localOffset) const;
    
    const Eigen::MatrixXd& getMassMatrix()	const
    {
        return (*this)->getMassMatrix();
    }

    const Eigen::VectorXd& getConstraintForces() const
    {
        return (*this)->getConstraintForces();
    }

    const Eigen::VectorXd& getCoriolisAndGravityForces() const
    {
        return (*this)->getCoriolisAndGravityForces();
    }

    std::size_t getNumJoints()	const
    {
        return (*this)->getNumJoints();
    }

    
    Joint* getDof_i_getJoint(int i)
    {
        return (*this)->getDof(i)->getJoint();
    }	

    
    Joint* getJoint(std::size_t	_idx)
    {
        return (*this)->getJoint(_idx);
    }	

    Eigen::Matrix4d getBodyNode_i_getTransform_mul_getParentJoint_getTransformFromChildBodyNode(int i)
    {
        Eigen::Isometry3d a = (*this)->getBodyNode(i)->getTransform() * (*this)->getBodyNode(i)->getParentJoint()->getTransformFromChildBodyNode();
        Eigen::Matrix4d b = Eigen::Matrix4d::Zero(4, 4);
        b.block<3, 3>(0, 0) = a.linear();
        b.block<3, 1>(0, 3) = a.translation();
        return b;
    }



};

#endif
