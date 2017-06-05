#pragma once

#include <vector>
#include <queue>
#include <algorithm>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>

struct RemesherTraits : OpenMesh::DefaultTraits
{
    VertexAttributes(OpenMesh::Attributes::Normal | OpenMesh::Attributes::Status);
    HalfedgeAttributes(OpenMesh::Attributes::PrevHalfedge | OpenMesh::Attributes::Status);
    EdgeAttributes(OpenMesh::Attributes::Status);
    FaceAttributes(OpenMesh::Attributes::Normal | OpenMesh::Attributes::Status);

    VertexTraits{
    public:
        Point cog;
    };
};

template <typename Mesh>
class Remesher {
public:
    static bool is_manifold(const Mesh &mesh) {
        return std::all_of(mesh.vertices_sbegin(), mesh.vertices_end(),
            [&](const Mesh::VertexHandle &vh) -> bool {
                return mesh.is_manifold(vh);
            }
        );
    }

    void run(Mesh &mesh, float target_length, int max_iter = 5, int max_smooth_iter = 5, float smooth_rate = 0.2) {
        if (!is_manifold(mesh)) {
            puts("Warning: mesh is not manifold!");
        }
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

    static void split_longer(Mesh &mesh, float target_length) {
        auto cmp = [&](const Mesh::EdgeHandle &eh1, const Mesh::EdgeHandle &eh2)->bool {
            return mesh.calc_edge_length(eh1) < mesh.calc_edge_length(eh2);
        };
        std::priority_queue<Mesh::EdgeHandle, std::vector<Mesh::EdgeHandle>, decltype(cmp)> edge_to_split(cmp);

        for (Mesh::EdgeIter ei = mesh.edges_sbegin(); ei != mesh.edges_end(); ++ei) {
            if (mesh.calc_edge_length(*ei) > target_length) {
                edge_to_split.emplace(*ei);
            }
        }

        while (!edge_to_split.empty()) {
            Mesh::EdgeHandle eh = edge_to_split.top();
            edge_to_split.pop();

            Mesh::HalfedgeHandle heh = mesh.halfedge_handle(eh, 0);
            Mesh::VertexHandle vh_from = mesh.from_vertex_handle(heh);
            Mesh::VertexHandle vh_to = mesh.to_vertex_handle(heh);
            Mesh::Point pt_from = mesh.point(vh_from);
            Mesh::Point pt_to = mesh.point(vh_to);

            Mesh::Point pt_mid = 0.5f*(pt_from + pt_to);
            Mesh::VertexHandle vh_mid = mesh.split(eh, pt_mid);

            for (Mesh::VertexEdgeCCWIter vei = mesh.ve_ccwbegin(vh_mid); vei != mesh.ve_ccwiter(vh_mid); ++vei) {
                if (mesh.calc_edge_length(*vei) > target_length) {
                    edge_to_split.emplace(*vei);
                }
            }
        }
    }


    static void collapse_shorter(Mesh &mesh, float target_length) {
        std::vector<Mesh::EdgeHandle> edge_to_collapse;
        for (Mesh::EdgeIter ei = mesh.edges_sbegin(); ei != mesh.edges_end(); ++ei) {
            if (mesh.calc_edge_length(*ei) < 0.8*target_length) {
                edge_to_collapse.emplace_back(*ei);
            }
        }

        for (Mesh::EdgeHandle &eh : edge_to_collapse) {
            Mesh::HalfedgeHandle heh = mesh.halfedge_handle(eh, 0);
            Mesh::VertexHandle vh_from = mesh.from_vertex_handle(heh);
            Mesh::VertexHandle vh_to = mesh.to_vertex_handle(heh);

            bool can_collapse = mesh.is_collapse_ok(heh) && !mesh.is_boundary(vh_from) && !mesh.is_boundary(vh_to) &&
                std::none_of(mesh.ve_ccwbegin(vh_from), mesh.ve_ccwend(vh_from), [&](const Mesh::EdgeHandle &eh) -> bool {
                return mesh.calc_edge_length(eh) >= 4.0*target_length / 3.0;
            });

            if (can_collapse) {
                mesh.collapse(heh);
            }
        }
    }

    static void adjust_valence(Mesh &mesh) {
        for (Mesh::EdgeIter ei = mesh.edges_sbegin(); ei != mesh.edges_end(); ++ei) {
            if (mesh.is_boundary(*ei) || !mesh.is_flip_ok(*ei)) continue;

            Mesh::HalfedgeHandle heh = mesh.halfedge_handle(*ei, 0);
            Mesh::VertexHandle a0 = mesh.to_vertex_handle(heh);
            Mesh::VertexHandle a1 = mesh.opposite_vh(heh);
            Mesh::VertexHandle a2 = mesh.from_vertex_handle(heh);
            Mesh::VertexHandle a3 = mesh.opposite_vh(mesh.opposite_halfedge_handle(heh));

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

            auto target_valence = [&](const Mesh::VertexHandle &vh) -> int {
                return mesh.is_boundary(vh) ? 4 : 6;
            };

            int diff_a0 = mesh.valence(a0) - target_valence(a0);
            int diff_a1 = mesh.valence(a1) - target_valence(a1);
            int diff_a2 = mesh.valence(a2) - target_valence(a2);
            int diff_a3 = mesh.valence(a3) - target_valence(a3);

            int dev_pre = abs(diff_a0) + abs(diff_a1) + abs(diff_a2) + abs(diff_a3);
            int dev_post = abs(diff_a0 - 1) + abs(diff_a1 + 1) + abs(diff_a2 - 1) + abs(diff_a3 + 1);

            if (dev_post < dev_pre) {
                mesh.flip(*ei);
            }
        }
    }

    static void smooth(Mesh &mesh, float lambda, bool tangential) {
        for (Mesh::VertexIter vi = mesh.vertices_sbegin(); vi != mesh.vertices_end(); ++vi) {
            if (mesh.is_boundary(*vi)) {
                mesh.data(*vi).cog = mesh.point(*vi);
            }
            else {
                mesh.data(*vi).cog = { 0.0f, 0.0f, 0.0f };
                for (Mesh::VertexVertexCCWIter vvi = mesh.vv_ccwbegin(*vi); vvi != mesh.vv_ccwend(*vi); ++vvi) {
                    mesh.data(*vi).cog += mesh.point(*vvi);
                }
                mesh.data(*vi).cog /= (float)mesh.valence(*vi);
                Mesh::Point shift = mesh.data(*vi).cog - mesh.point(*vi);
                if (tangential) {
                    Mesh::Normal normal = mesh.normal(*vi);
                    shift -= normal*dot(normal, shift);
                }
                mesh.data(*vi).cog = mesh.point(*vi) + lambda*shift;
            }
        }
        for (Mesh::VertexIter vi = mesh.vertices_sbegin(); vi != mesh.vertices_end(); ++vi) {
            mesh.set_point(*vi, mesh.data(*vi).cog);
        }
    }
};
