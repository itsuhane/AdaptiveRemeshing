#include <iostream>
#include <OpenMesh/Core/IO/MeshIO.hh>
#include "Remesher.h"

int main(int argc, char *argv[]) {
    typedef OpenMesh::TriMesh_ArrayKernelT<RemesherTraits> Mesh;
    Mesh mesh;
    OpenMesh::IO::read_mesh(mesh, argv[1]);
    Remesher<Mesh> remesher;
    remesher.run(mesh, 0.001f);
    OpenMesh::IO::write_mesh(mesh, argv[2]);
    return 0;
}
