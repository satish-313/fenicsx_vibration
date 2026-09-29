import gmsh
import sys

part_path = ""
if len(sys.argv) > 1:
    part_path = sys.argv[1]


gmsh.initialize()
gmsh.model.add("test")
gmsh.model.occ.importShapes(f"./cad/{part_path}.step")
gmsh.model.occ.synchronize()

volumes = gmsh.model.occ.getEntities(dim=3)
print("original vols : ",len(volumes))
surfaces = gmsh.model.occ.getEntities(dim=2)
print("original surfaces : ",len(surfaces))

gmsh.model.occ.fragment(volumes,[])
gmsh.model.occ.removeAllDuplicates()
gmsh.model.occ.synchronize()

new_volumes = gmsh.model.occ.getEntities(dim=3)
print("new volumes : ",len(new_volumes))
new_surface = gmsh.model.occ.getEntities(dim=2)
print("new surfaces : ",len(new_surface))

gmsh.finalize()
