#include <iostream>
#include <OpenMesh/Core/IO/MeshIO.hh>
#include "Remesher.h"

std::vector<size_t> print_histogram(const std::vector<float> &values, size_t n_bins = 10) {
    auto m = std::minmax_element(values.begin(), values.end());
    float lower_bound = *m.first;
    float upper_bound = *m.second;

    std::vector<size_t> histogram(n_bins, 0);

    for (auto &v : values) {
        size_t bin_id = std::min(size_t(n_bins*(v - lower_bound) / (upper_bound - lower_bound)), n_bins-1);
        histogram[bin_id]++;
    }

    size_t scale = *std::max_element(histogram.begin(), histogram.end());

    int len = int(log10(scale)) + 1;

    for (int i = (int)histogram.size()-1; i >=0 ; --i) {
        printf("%+.2e ~ %+.2e : [%*zd] | ", lower_bound + i * (upper_bound - lower_bound) / n_bins, lower_bound + (i + 1) * (upper_bound - lower_bound) / n_bins, len, histogram[i]);
        for (size_t j = 0; j < 50 * histogram[i] / scale; ++j) {
            printf("*");
        }
        puts("");
    }

    return histogram;
}

int main(int argc, char *argv[]) {
    typedef OpenMesh::TriMesh_ArrayKernelT<RemesherTraits> Mesh;
    Mesh mesh;
    OpenMesh::IO::read_mesh(mesh, argv[1]);

    CurvatureEstimator<Mesh>::run(mesh);
    std::vector<float> kmeans;
    for (Mesh::VertexIter vi = mesh.vertices_sbegin(); vi != mesh.vertices_end(); ++vi) {
        float abs_kmean = abs(mesh.data(*vi).kmean());
        kmeans.push_back(abs_kmean);
    }
    print_histogram(kmeans,25);

    mesh.request_vertex_colors();
    float kbound = *std::max_element(kmeans.begin(), kmeans.end());
    for (Mesh::VertexIter vi = mesh.vertices_sbegin(); vi != mesh.vertices_end(); ++vi) {
        float abs_kmean = abs(mesh.data(*vi).kmean());
        if (abs_kmean > 0.04*kbound) {
            mesh.set_color(*vi, { 255,0,0 });
        }
        else if (abs_kmean > 0.02*kbound) {
            mesh.set_color(*vi, { 0,255,0 });
        }
        else {
            mesh.set_color(*vi, { 0,0,255 });
        }
    }

    //    Remesher<Mesh>::run(mesh);

    OpenMesh::IO::write_mesh(mesh, argv[2], OpenMesh::IO::Options::VertexColor);
    return 0;
}
