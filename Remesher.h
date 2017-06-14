#pragma once

#include <vector>
#include <queue>
#include <algorithm>
#include <chrono>
#include <map>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include "CurvatureEstimator.h"
#include "maxflow/graph.h"

struct RemesherTraits : OpenMesh::DefaultTraits
{
    VertexAttributes(OpenMesh::Attributes::Normal | OpenMesh::Attributes::Status);
    HalfedgeAttributes(OpenMesh::Attributes::PrevHalfedge | OpenMesh::Attributes::Status);
    EdgeAttributes(OpenMesh::Attributes::Status);
    FaceAttributes(OpenMesh::Attributes::Normal | OpenMesh::Attributes::Status);

    template <class Base, class Refs> struct VertexT : public CurvatureEstimatorTraits::VertexT<Base, Refs> {
    public:
        bool is_locked() const {
            return m_locked;
        }
        void set_locked(bool locked) {
            m_locked = locked;
        }

        float weight() const {
            return m_weight;
        }
        void set_weight(float weight) {
            m_weight = weight;
        }
    private:
        bool m_locked = false;
        float m_weight = 0.0f;
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

    FaceTraits {
    public:
        float weight() const {
            return m_weight;
        }
        void set_weight(float weight) {
            m_weight = weight;
        }
    private:
        float m_weight = 0.0;
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
    typedef typename Mesh::VertexOHalfedgeCCWIter VertexOHalfedgeCCWIter;

    typedef typename Mesh::EdgeHandle EdgeHandle;
    typedef typename Mesh::EdgeIter EdgeIter;

    typedef typename Mesh::FaceHandle FaceHandle;
    typedef typename Mesh::FaceIter FaceIter;
    typedef typename Mesh::FaceVertexCCWIter FaceVertexCCWIter;

    typedef typename Mesh::HalfedgeHandle HalfedgeHandle;

    typedef Graph<Scalar, Scalar, Scalar> MaxFlowGraph;

    static bool is_manifold(const Mesh &mesh) {
        return std::all_of(mesh.vertices_sbegin(), mesh.vertices_end(),
            [&](const VertexHandle &vh) -> bool {
                return mesh.is_manifold(vh);
            }
        );
    }

    static Scalar estimate_target_length(const Mesh &mesh, Scalar lambda = 0.5) {
        printf("estimate_target_length...");
        std::vector<Scalar> lengths;
        for (EdgeIter ei = mesh.edges_sbegin(); ei != mesh.edges_end(); ++ei) {
            lengths.push_back(mesh.calc_edge_length(*ei));
        }
        std::sort(lengths.begin(), lengths.end());
        puts("done.");
        return lengths[size_t((lengths.size() - 1)*lambda)];
    }

    static void run(Mesh &mesh, Scalar target_length, int max_iter = 5, int max_smooth_iter = 5, Scalar smooth_rate = 0.2) {
        if (!is_manifold(mesh)) {
            puts("Warning: mesh is not manifold!");
        }
        for (int i = max_iter; i > 0; --i) {
            split_longer(mesh, target_length);
            collapse_shorter(mesh, target_length);
            adjust_valence(mesh);
            clean_triangles(mesh);
            printf("smoothing...");
            for (int j = 0; j < max_smooth_iter; ++j) {
                smooth(mesh, smooth_rate, i > 1);
            }
            puts("done.");
        }
    }

    static void run_bicut(
        Mesh &mesh,
        Scalar dense_length,
        Scalar sparse_length,
        Scalar dense_percentile = 0.5f,
        Scalar sparse_percentile = 0.6f,
        Scalar gc_weight = 1.25f,
        int max_iter = 5,
        int max_smooth_iter = 5,
        Scalar smooth_rate = 0.2
    ) {
        Remesher<Mesh>::run(mesh, dense_length, max_iter, max_smooth_iter, smooth_rate);
        printf("estimating curvature...");
        CurvatureEstimator<Mesh>::run(mesh);
        puts("done.");
        Remesher<Mesh>::segment_feature_area(mesh, dense_percentile, sparse_percentile, gc_weight);
        Remesher<Mesh>::lock_feature_area(mesh);
        Remesher<Mesh>::run(mesh, sparse_length, max_iter, max_smooth_iter, smooth_rate);
    }

    static void split_longer(Mesh &mesh, Scalar target_length) {
        puts("split_longer...");

        puts("  listing edges...");
        auto cmp = [&](const EdgeHandle &eh1, const EdgeHandle &eh2)->bool {
            return mesh.calc_edge_length(eh1) < mesh.calc_edge_length(eh2);
        };
        std::priority_queue<EdgeHandle, std::vector<EdgeHandle>, decltype(cmp)> edge_to_split(cmp);
        for (EdgeIter ei = mesh.edges_sbegin(); ei != mesh.edges_end(); ++ei) {
            if (mesh.calc_edge_length(*ei) > target_length) {
                edge_to_split.emplace(*ei);
            }
        }

        using std::chrono::seconds;
        using std::chrono::system_clock;

        puts("  splitting...");
        size_t nv = mesh.n_vertices(), ne = mesh.n_edges(), nf = mesh.n_faces();
        system_clock::time_point T0 = system_clock::now();
        printf("% 80s\r    (V, E, F) = (%zd, %zd, %zd) -> (%zd, %zd, %zd)\r", "", nv, ne, nf, mesh.n_vertices(), mesh.n_edges(), mesh.n_faces());
        while (!edge_to_split.empty()) {
            system_clock::time_point T1 = system_clock::now();
            if (std::chrono::duration_cast<seconds>(T1-T0).count() >= 1) {
                T0 = T1;
                printf("% 80s\r    (V, E, F) = (%zd, %zd, %zd) -> (%zd, %zd, %zd)\r", "", nv, ne, nf, mesh.n_vertices(), mesh.n_edges(), mesh.n_faces());
            }

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
        printf("% 80s\r    (V, E, F) = (%zd, %zd, %zd) -> (%zd, %zd, %zd)\n  done.\n", "", nv, ne, nf, mesh.n_vertices(), mesh.n_edges(), mesh.n_faces());
    }


    static void collapse_shorter(Mesh &mesh, Scalar target_length) {
        puts("collapse_shorter...");

        puts("  listing edges...");
        std::vector<EdgeHandle> edge_to_collapse;
        for (EdgeIter ei = mesh.edges_sbegin(); ei != mesh.edges_end(); ++ei) {
            if (mesh.data(*ei).is_feature()) continue; // feature edge cannot collapse
            if (mesh.calc_edge_length(*ei) < 0.8*target_length) {
                edge_to_collapse.emplace_back(*ei);
            }
        }

        using std::chrono::seconds;
        using std::chrono::system_clock;

        puts("  collapsing...");
        char progress_char[] = { '\\', '|', '/', '-' };
        size_t nv = mesh.n_vertices(), ne = mesh.n_edges(), nf = mesh.n_faces(), pr = 0;
        system_clock::time_point T0 = system_clock::now();
        printf("% 80s\r    (V, E, F) = (%zd, %zd, %zd) -> \r", "", nv, ne, nf);
        for (EdgeHandle &eh : edge_to_collapse) {
            for (int side = 0; side <= 1; ++side) {
                system_clock::time_point T1 = system_clock::now();
                if (std::chrono::duration_cast<seconds>(T1 - T0).count() >= 1) {
                    T0 = T1;
                    printf("% 80s\r    (V, E, F) = (%zd, %zd, %zd) %c> \r", "", nv, ne, nf, progress_char[pr]);
                    pr = (pr + 1) % 4;
                }

                HalfedgeHandle heh = mesh.halfedge_handle(eh, side);
                VertexHandle vh_from = mesh.from_vertex_handle(heh);
                VertexHandle vh_to = mesh.to_vertex_handle(heh);

                bool topo_can_collapse = !mesh.is_boundary(vh_from) // boundary vertex cannot move
                    && !mesh.data(vh_from).is_locked() // locked vertex cannot move
                    && mesh.is_collapse_ok(heh) // collapse cannot harm topology
                    && std::none_of(mesh.ve_ccwbegin(vh_from), mesh.ve_ccwend(vh_from), [&](const EdgeHandle &eh) -> bool { // collapse cannot introduce long edge
                        return mesh.calc_edge_length(eh) >= 4.0f*target_length / 3.0f;
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
                    if (OpenMesh::dot(OpenMesh::cross(vprev, vto), normal) < 0.0f) {
                        geom_can_collapse = false;
                        break;
                    }
                }
                if (!geom_can_collapse) continue;

                mesh.collapse(heh);
            }
        }
        mesh.garbage_collection();
        printf("% 80s\r    (V, E, F) = (%zd, %zd, %zd) -> (%zd, %zd, %zd)\n  done.\n", "", nv, ne, nf, mesh.n_vertices(), mesh.n_edges(), mesh.n_faces());
    }

    static void adjust_valence(Mesh &mesh) {
        using std::chrono::seconds;
        using std::chrono::system_clock;

        size_t ne = mesh.n_edges(), ncount = 0;
        system_clock::time_point T0 = system_clock::now();
        printf("% 80s\radjust_valence...%zd/%zd\r", "", ncount, ne);
        for (EdgeIter ei = mesh.edges_sbegin(); ei != mesh.edges_end(); ++ei) {
            ncount++;
            system_clock::time_point T1 = system_clock::now();
            if (std::chrono::duration_cast<seconds>(T1 - T0).count() >= 1) {
                T0 = T1;
                printf("% 80s\radjust_valence...%zd/%zd\r", "", ncount, ne);
            }
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
        printf("% 80s\radjust_valence...done.\n", "");
    }

    static void smooth(Mesh &mesh, Scalar lambda, bool tangential) {
        mesh.update_normals();
        OpenMesh::VPropHandleT<Point> cog_p;
        mesh.add_property(cog_p);
        for (VertexIter vi = mesh.vertices_sbegin(); vi != mesh.vertices_end(); ++vi) {
            if (mesh.is_boundary(*vi) || mesh.data(*vi).is_locked()) {
                mesh.property(cog_p, *vi) = mesh.point(*vi);
            }
            else {
                mesh.property(cog_p, *vi) = { 0.0f, 0.0f, 0.0f };
                for (VertexVertexCCWIter vvi = mesh.vv_ccwbegin(*vi); vvi != mesh.vv_ccwend(*vi); ++vvi) {
                    mesh.property(cog_p, *vi) += mesh.point(*vvi);
                }
                mesh.property(cog_p, *vi) /= (Scalar)mesh.valence(*vi);
                Point shift = mesh.property(cog_p, *vi) - mesh.point(*vi);
                if (tangential) {
                    Normal normal = mesh.normal(*vi);
                    shift -= normal*dot(normal, shift);
                }
                mesh.property(cog_p, *vi) = mesh.point(*vi) + lambda*shift;
            }
        }
        for (VertexIter vi = mesh.vertices_sbegin(); vi != mesh.vertices_end(); ++vi) {
            mesh.set_point(*vi, mesh.property(cog_p, *vi));
        }
        mesh.remove_property(cog_p);
    }

    static void clean_triangles(Mesh &mesh) {
        printf("clean_triangles...");
        for (VertexIter vi = mesh.vertices_sbegin(); vi != mesh.vertices_end(); ++vi) {
            if (mesh.data(*vi).is_locked() || mesh.is_boundary(*vi)) continue;
            if (mesh.valence(*vi) == 3) {
                for (VertexOHalfedgeCCWIter vohei = mesh.voh_ccwbegin(*vi); vohei != mesh.voh_ccwend(*vi); ++vohei) {
                    if (mesh.is_collapse_ok(*vohei)) {
                        mesh.collapse(*vohei);
                        break;
                    }
                }
            }
        }
        mesh.garbage_collection();
        puts("done.");
    }

    static void segment_feature_area(Mesh &mesh, Scalar lambda_1, Scalar lambda_2, Scalar weight) {
        puts("segment_feature_area...");
        mesh.garbage_collection();

        puts("  building graph...");
        if (lambda_1 > lambda_2) std::swap(lambda_1, lambda_2);

        std::map<Scalar, Scalar> order_cost;
        for (VertexIter vi = mesh.vertices_sbegin(); vi != mesh.vertices_end(); ++vi) {
            order_cost.emplace(abs(mesh.data(*vi).kmean()), 0.0f);
        }
        Scalar counter = -1.0f;
        for (auto &c : order_cost) {
            counter += 1.0f;
            c.second = counter;
        }
        for (auto &c : order_cost) {
            c.second = c.second / counter;
        }

        MaxFlowGraph graph((int)mesh.n_faces(), (int)mesh.n_edges());
        graph.add_node((int)mesh.n_faces());

        auto smootherstep = [](Scalar l, Scalar r, Scalar x) -> Scalar {
            x = std::min(std::max((x - l) / (r - l), Scalar(0)), Scalar(1));
            return x*x*x*(x*(x * 6 - 15) + 10);
        };

        for (FaceIter fi = mesh.faces_sbegin(); fi != mesh.faces_end(); ++fi) {
            Scalar order_cost_avg = 0.0f;
            int count = 0;
            for (FaceVertexCCWIter fvi = mesh.fv_ccwbegin(*fi); fvi != mesh.fv_ccwend(*fi); ++fvi) {
                order_cost_avg += order_cost[abs(mesh.data(*fvi).kmean())];
                count++;
            }
            order_cost_avg /= count;
            HalfedgeHandle heh = mesh.halfedge_handle(*fi);
            Scalar area = mesh.calc_sector_area(heh);
            Scalar w = smootherstep(lambda_1, lambda_2, order_cost_avg);
            graph.add_tweights(fi->idx(), area*w, area*(1 - w));
            mesh.data(*fi).set_weight(w);
        }

        for (EdgeIter ei = mesh.edges_sbegin(); ei != mesh.edges_end(); ++ei) {
            if (mesh.is_boundary(*ei)) continue;
            HalfedgeHandle heh = mesh.halfedge_handle(*ei, 0);
            FaceHandle fh0 = mesh.face_handle(heh);
            FaceHandle fh1 = mesh.opposite_face_handle(heh);
            Scalar len = mesh.calc_edge_length(heh)*weight;
            graph.add_edge(fh0.idx(), fh1.idx(), len, len);
        }

        puts("  computing max-flow...");
        graph.maxflow();

        for (FaceIter fi = mesh.faces_sbegin(); fi != mesh.faces_end(); ++fi) {
            if (graph.what_segment(fi->idx()) == MaxFlowGraph::SOURCE) {
                mesh.data(*fi).set_weight(1.0f);
            }
            else {
                mesh.data(*fi).set_weight(0.0f);
            }
        }
        puts("  done.");
    }

    static void lock_feature_area(Mesh &mesh) {
        for (Mesh::VertexIter vi = mesh.vertices_sbegin(); vi != mesh.vertices_end(); ++vi) {
            for (Mesh::VertexFaceCCWIter vfi = mesh.vf_ccwbegin(*vi); vfi != mesh.vf_ccwend(*vi); ++vfi) {
                if (mesh.data(*vfi).weight() > 0.5f) {
                    mesh.data(*vi).set_locked(true);
                }
            }
        }
    }
};
