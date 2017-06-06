#pragma once

#include <array>
#include <Eigen/Eigen>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>

struct CurvatureEstimatorTraits : OpenMesh::DefaultTraits
{
    VertexAttributes(OpenMesh::Attributes::Normal | OpenMesh::Attributes::Status);
    HalfedgeAttributes(OpenMesh::Attributes::PrevHalfedge | OpenMesh::Attributes::Status);
    EdgeAttributes(OpenMesh::Attributes::Status);
    FaceAttributes(OpenMesh::Attributes::Normal | OpenMesh::Attributes::Status);

    VertexTraits {
    public:
        float k1;
        float k2;

        Normal d1;
        Normal d2;

        float kmax() const {
            return k1;
        }

        float kmin() const {
            return k2;
        }

        float kmean() const {
            return 0.5f*(k1 + k2);
        }

        const Normal &direction_kmax() const {
            return d1;
        }

        const Normal &direction_kmin() const {
            return d2;
        }
    };
};

template <typename Mesh>
class CurvatureEstimator {
public:
    typedef typename Mesh::Scalar Scalar;
    typedef typename Mesh::Normal Normal;

    typedef typename Mesh::VertexHandle VertexHandle;
    typedef typename Mesh::VertexIter VertexIter;
    typedef typename Mesh::VertexOHalfedgeCCWIter VertexOHalfedgeCCWIter;

    typedef typename Mesh::HalfedgeHandle HalfedgeHandle;

    static void run(Mesh &mesh) {
        mesh.update_normals();
        for (VertexIter vi = mesh.vertices_sbegin(); vi != mesh.vertices_end(); ++vi) {
            estimate_vertex_curvature(mesh, *vi);
        }
    }

    static void estimate_vertex_curvature(Mesh &mesh, const VertexHandle &vh) {
        typedef Eigen::Matrix<Scalar, 3, 3> Matrix3S;
        Matrix3S tensor3;
        tensor3.setZero();
        for (VertexOHalfedgeCCWIter vohei = mesh.voh_ccwbegin(vh); vohei != mesh.voh_ccwend(vh); ++vohei) {
            HalfedgeHandle ohe = *vohei;
            HalfedgeHandle ihe = mesh.opposite_halfedge_handle(ohe);

            if (mesh.is_boundary(ohe) || mesh.is_boundary(ihe)) continue;

            Normal edge_vector = mesh.calc_edge_vector(ohe);
            Scalar edge_length = edge_vector.norm();
            edge_vector.normalize_cond(); // edge_vector is normalized!

            Normal normal1 = mesh.normal(mesh.face_handle(ohe));
            Normal normal2 = mesh.normal(mesh.face_handle(ihe));

            Scalar sinus = OpenMesh::dot(OpenMesh::cross(normal1, normal2), edge_vector);
            Scalar beta = asin(OpenMesh::sane_aarg(sinus));

            Eigen::Matrix<Scalar, 3, 1> eev{ edge_vector[0], edge_vector[1], edge_vector[2] };

            tensor3 += beta * edge_length * eev * eev.transpose();
        }

        Eigen::SelfAdjointEigenSolver<Matrix3S> solver(tensor3);

        // reorder eigenvalues according to their absolute value.
        std::array<int, 3> order{ 0, 1, 2 };
        std::array<Scalar, 3> abs_lambda;
        abs_lambda[0] = abs(solver.eigenvalues()[0]);
        abs_lambda[1] = abs(solver.eigenvalues()[1]);
        abs_lambda[2] = abs(solver.eigenvalues()[2]);
        if (abs_lambda[0] <= abs_lambda[1] && abs_lambda[0] <= abs_lambda[2]) {
            order[0] = 0;
            if (abs_lambda[1] <= abs_lambda[2]) {
                order[1] = 1;
                order[2] = 2;
            }
            else {
                order[1] = 2;
                order[2] = 1;
            }
        }
        else if (abs_lambda[1] <= abs_lambda[0] && abs_lambda[1] <= abs_lambda[2]) {
            order[0] = 1;
            if (abs_lambda[0] <= abs_lambda[2]) {
                order[1] = 0;
                order[2] = 2;
            }
            else {
                order[1] = 2;
                order[2] = 0;
            }
        }
        else {
            order[0] = 2;
            if (abs_lambda[0] <= abs_lambda[1]) {
                order[1] = 0;
                order[2] = 1;
            }
            else {
                order[1] = 1;
                order[2] = 0;
            }
        }

        Matrix3S V = solver.eigenvectors();
        Normal d1{ V(0, order[2]), V(1, order[2]) , V(2, order[2]) };
        Normal d2{ V(0, order[1]), V(1, order[1]) , V(2, order[1]) };

        mesh.data(vh).d1 = d1;
        mesh.data(vh).d2 = d2;
        mesh.data(vh).k1 = solver.eigenvalues()[order[2]];
        mesh.data(vh).k2 = solver.eigenvalues()[order[1]];
    }
};
