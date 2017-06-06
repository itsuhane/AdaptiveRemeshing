#pragma once

#include <vector>
#include <queue>
#include <algorithm>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include "CurvatureEstimator.h"

struct RemesherTraits : OpenMesh::DefaultTraits
{
    VertexAttributes(OpenMesh::Attributes::Normal | OpenMesh::Attributes::Status);
    HalfedgeAttributes(OpenMesh::Attributes::PrevHalfedge | OpenMesh::Attributes::Status);
    EdgeAttributes(OpenMesh::Attributes::Status);
    FaceAttributes(OpenMesh::Attributes::Normal | OpenMesh::Attributes::Status);

    template <class Base, class Refs> struct VertexT : public CurvatureEstimatorTraits::VertexT<Base, Refs> {
    public:
        Point cog;

        bool is_locked() const {
            return m_locked;
        }
        void set_locked(bool locked) {
            m_locked = locked;
        }
    private:
        bool m_locked = false;
    };

    EdgeTraits {
    public:
        bool is_feature() const {
            return m_feature;
        }
        void set_feature(bool feature) {
            m_feature = feature;
        }
    private:
        bool m_feature = false;
    };
};

template <typename Mesh>
class Remesher {
public:
    typedef typename Mesh::Scalar Scalar;
    typedef typename Mesh::Point Point;
    typedef typename Mesh::Normal Normal;

    typedef typename Mesh::VertexHandle VertexHandle;
    typedef typename Mesh::VertexIter VertexIter;
    typedef typename Mesh::VertexVertexCCWIter VertexVertexCCWIter;
    typedef typename Mesh::VertexEdgeCCWIter VertexEdgeCCWIter;

    typedef typename Mesh::EdgeHandle EdgeHandle;
    typedef typename Mesh::EdgeIter EdgeIter;

    typedef typename Mesh::HalfedgeHandle HalfedgeHandle;

    static bool is_manifold(const Mesh &mesh) {
        return std::all_of(mesh.vertices_sbegin(), mesh.vertices_end(),
            [&](const VertexHandle &vh) -> bool {
                return mesh.is_manifold(vh);
            }
        );
    }

    static Scalar estimate_target_length(const Mesh &mesh, const Scalar lambda = 0.5) {
        std::vector<Scalar> lengths;
        for (EdgeIter ei = mesh.edges_sbegin(); ei != mesh.edges_end(); ++ei) {
            lengths.push_back(mesh.calc_edge_length(*ei));
        }
        std::sort(lengths.begin(), lengths.end());
        return lengths[size_t((lengths.size() - 1)*lambda)];
    }

    static void run(Mesh &mesh, Scalar target_length_lambda = 0.5, int max_iter = 5, int max_smooth_iter = 5, Scalar smooth_rate = 0.2) {
        if (!is_manifold(mesh)) {
            puts("Warning: mesh is not manifold!");
        }
        Scalar target_length = estimate_target_length(mesh, target_length_lambda);
        for (int i = max_iter; i > 0; --i) {
            split_longer(mesh, target_length);
            collapse_shorter(mesh, target_length);
            adjust_valence(mesh);
            mesh.garbage_collection();
            for (int j = 0; j < max_smooth_iter; ++j) {
                mesh.update_normals();
                smooth(mesh, smooth_rate, i > 1);
            }
        }
    }

    static void split_longer(Mesh &mesh, Scalar target_length) {
        auto cmp = [&](const EdgeHandle &eh1, const EdgeHandle &eh2)->bool {
            return mesh.calc_edge_length(eh1) < mesh.calc_edge_length(eh2);
        };
        std::priority_queue<EdgeHandle, std::vector<EdgeHandle>, decltype(cmp)> edge_to_split(cmp);

        for (EdgeIter ei = mesh.edges_sbegin(); ei != mesh.edges_end(); ++ei) {
            if (mesh.calc_edge_length(*ei) > target_length) {
                edge_to_split.emplace(*ei);
            }
        }

        while (!edge_to_split.empty()) {
            EdgeHandle eh = edge_to_split.top();
            edge_to_split.pop();

            bool is_feature = mesh.data(eh).is_feature();

            HalfedgeHandle heh = mesh.halfedge_handle(eh, 0);
            VertexHandle vh_from = mesh.from_vertex_handle(heh);
            VertexHandle vh_to = mesh.to_vertex_handle(heh);
            Point pt_from = mesh.point(vh_from);
            Point pt_to = mesh.point(vh_to);

            Point pt_mid = 0.5f*(pt_from + pt_to);
            VertexHandle vh_mid = mesh.split(eh, pt_mid);

            for (VertexEdgeCCWIter vei = mesh.ve_ccwbegin(vh_mid); vei != mesh.ve_ccwend(vh_mid); ++vei) {
                HalfedgeHandle heh = mesh.halfedge_handle(*vei, 0);
                VertexHandle vh = mesh.to_vertex_handle(heh);
                if (vh == vh_from || vh == vh_to) {
                    mesh.data(*vei).set_feature(is_feature); // feature can be split, but must retain its feature flag.
                }
                if (mesh.calc_edge_length(*vei) > target_length) {
                    edge_to_split.emplace(*vei);
                }
            }
        }
    }


    static void collapse_shorter(Mesh &mesh, Scalar target_length) {
        std::vector<EdgeHandle> edge_to_collapse;
        for (EdgeIter ei = mesh.edges_sbegin(); ei != mesh.edges_end(); ++ei) {
            if (mesh.data(*ei).is_feature()) continue; // feature edge cannot collapse
            if (mesh.calc_edge_length(*ei) < 0.8*target_length) {
                edge_to_collapse.emplace_back(*ei);
            }
        }

        for (EdgeHandle &eh : edge_to_collapse) {
            for (int side = 0; side <= 1; ++side) {
                HalfedgeHandle heh = mesh.halfedge_handle(eh, side);
                VertexHandle vh_from = mesh.from_vertex_handle(heh);
                VertexHandle vh_to = mesh.to_vertex_handle(heh);

                bool topo_can_collapse = !mesh.is_boundary(vh_from) // boundary vertex cannot move
                    && !mesh.data(vh_from).is_locked() // locked vertex cannot move
                    && mesh.is_collapse_ok(heh) // collapse cannot harm topology
                    && std::none_of(mesh.ve_ccwbegin(vh_from), mesh.ve_ccwend(vh_from), [&](const EdgeHandle &eh) -> bool { // collapse cannot introduce long edge
                        return mesh.calc_edge_length(eh) >= 4.0*target_length / 3.0;
                    }
                );

                if (!topo_can_collapse) continue;

                // topologically ok to collapse
                // now check geometry constraints
                // to vertex must see all 1-ring neighbors of from vertex
                Normal normal;
                mesh.calc_vertex_normal_loop(vh_from, normal);

                Point pto = mesh.point(vh_to);
                std::vector<Point> ring;
                for (HalfedgeHandle rheh = mesh.cw_rotated_halfedge_handle(heh); rheh != heh; rheh = mesh.cw_rotated_halfedge_handle(rheh)) {
                    ring.push_back(mesh.point(mesh.to_vertex_handle(rheh)));
                }

                bool geom_can_collapse = true;
                for (size_t i = 1; i < ring.size(); ++i) {
                    Point vto = pto - ring[i];
                    Point vprev = ring[i - 1] - ring[i];
                    if (OpenMesh::dot(OpenMesh::cross(vprev, vto), normal) < 0) {
                        geom_can_collapse = false;
                        break;
                    }
                }
                if (!geom_can_collapse) continue;

                mesh.collapse(heh);
            }
        }
    }

    static void adjust_valence(Mesh &mesh) {
        for (EdgeIter ei = mesh.edges_sbegin(); ei != mesh.edges_end(); ++ei) {
            if (mesh.is_boundary(*ei) || mesh.data(*ei).is_feature() || !mesh.is_flip_ok(*ei)) continue;

            HalfedgeHandle heh = mesh.halfedge_handle(*ei, 0);
            VertexHandle a0 = mesh.to_vertex_handle(heh);
            VertexHandle a1 = mesh.opposite_vh(heh);
            VertexHandle a2 = mesh.from_vertex_handle(heh);
            VertexHandle a3 = mesh.opposite_vh(mesh.opposite_halfedge_handle(heh));

            /*
                     a1
                     +
                 /       \
                /         \
            a2 +---------->+ a0
                \   heh   /
                 \       /
                     +
                     a3
            */

            auto target_valence = [&](const VertexHandle &vh) -> int {
                return mesh.is_boundary(vh) ? 4 : 6;
            };

            int diff_a0 = mesh.valence(a0) - target_valence(a0);
            int diff_a1 = mesh.valence(a1) - target_valence(a1);
            int diff_a2 = mesh.valence(a2) - target_valence(a2);
            int diff_a3 = mesh.valence(a3) - target_valence(a3);

            int dev_pre = abs(diff_a0) + abs(diff_a1) + abs(diff_a2) + abs(diff_a3);
            int dev_post = abs(diff_a0 - 1) + abs(diff_a1 + 1) + abs(diff_a2 - 1) + abs(diff_a3 + 1);

            if (dev_post < dev_pre) {
                // topologically it is preferred to flip
                // but we have to pay attention that a flip is geometrically ok
                Point p0 = mesh.point(a0);
                Point p1 = mesh.point(a1);
                Point p2 = mesh.point(a2);
                Point p3 = mesh.point(a3);

                Point v20 = (p0 - p2).normalize_cond();
                Point v02 = -v20;

                Point v01 = (p1 - p0).normalize_cond();
                Point v03 = (p3 - p0).normalize_cond();

                Point v21 = (p1 - p2).normalize_cond();
                Point v23 = (p3 - p2).normalize_cond();

                Scalar angle0 = (acos(OpenMesh::sane_aarg(OpenMesh::dot(v01, v02))) + acos(OpenMesh::sane_aarg(OpenMesh::dot(v03, v02))));
                Scalar angle2 = (acos(OpenMesh::sane_aarg(OpenMesh::dot(v21, v20))) + acos(OpenMesh::sane_aarg(OpenMesh::dot(v23, v20))));

                if (angle0 < 0.9*M_PI && angle2 < 0.9*M_PI) {
                    mesh.flip(*ei);
                }
            }
        }
    }

    static void smooth(Mesh &mesh, Scalar lambda, bool tangential) {
        for (VertexIter vi = mesh.vertices_sbegin(); vi != mesh.vertices_end(); ++vi) {
            if (mesh.is_boundary(*vi) || mesh.data(*vi).is_locked()) {
                mesh.data(*vi).cog = mesh.point(*vi);
            }
            else {
                mesh.data(*vi).cog = { 0.0f, 0.0f, 0.0f };
                for (VertexVertexCCWIter vvi = mesh.vv_ccwbegin(*vi); vvi != mesh.vv_ccwend(*vi); ++vvi) {
                    mesh.data(*vi).cog += mesh.point(*vvi);
                }
                mesh.data(*vi).cog /= (Scalar)mesh.valence(*vi);
                Point shift = mesh.data(*vi).cog - mesh.point(*vi);
                if (tangential) {
                    Normal normal = mesh.normal(*vi);
                    shift -= normal*dot(normal, shift);
                }
                mesh.data(*vi).cog = mesh.point(*vi) + lambda*shift;
            }
        }
        for (VertexIter vi = mesh.vertices_sbegin(); vi != mesh.vertices_end(); ++vi) {
            mesh.set_point(*vi, mesh.data(*vi).cog);
        }
    }
};
