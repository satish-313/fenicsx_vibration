import dolfinx
print(dolfinx.__version__)

from dolfinx import fem, io
import numpy as np
from mpi4py import MPI
import gmsh
import ufl
from slepc4py import SLEPc
from dolfinx.fem import petsc
from dolfinx.io import gmsh as gmshio
from petsc4py import PETSc

# gmsh
gmsh.initialize()
gmsh.model.add("cubesat")
gmsh.model.occ.importShapes("./cubesat_main.step")
gmsh.model.occ.synchronize()

# all solids 3d
solids = gmsh.model.getEntities(dim=3)
print("Solids: ", len(solids))

# fragment operation in gmsh to conformal meshing
gmsh.model.occ.fragment(solids, [])
gmsh.model.occ.removeAllDuplicates()
gmsh.model.occ.synchronize()

aluminim_tags = [1,2,3,4,23,24,25,26, 45,46,47,48,85]

all_volumes_tags = gmsh.model.occ.getEntities(3)

steel_tags = [tag for dim,tag in all_volumes_tags if tag not in aluminim_tags]

gmsh.model.addPhysicalGroup(3,aluminim_tags,tag=1,name="aluminium")
gmsh.model.addPhysicalGroup(3,steel_tags,tag=2,name="steel")

# clamped surfaces
clamped_surfaces = [1172, 1173, 1174, 1175, 1176, 1177, 1179]
gmsh.model.addPhysicalGroup(2, clamped_surfaces, tag=10)
gmsh.model.setPhysicalName(2, 10, "Clamped_Base")


# proper messhing parameters
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature",1)
gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi",10)

gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5)
gmsh.option.setNumber("Mesh.MeshSizeMax", 4.0)

gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

# mesh generation
try:
    gmsh.model.mesh.generate(3)
except Exception as e:
   print(e)

gmsh.write("cubesat.msh")



# verifying tags
tol = 1e-3 
aluminium_parts_point = [
    dict(point=(1.0,1.0,1.0), name="base_stack"),
    dict(point=(0.0,0.0,11.0),name="ll_base_cover"),
    dict(point=(8.0,0.0,25.0),name="l_casette"),
    dict(point=(0.0,0.0,39.0),name="lt_base_cover"),
    dict(point=(0.0,0.0,47.0),name="middle_stack"),
    dict(point=(0.0,0.0,56.0),name="lm_base_cover"),
    dict(point=(8.0,0.0,65.0),name="m_casette"),
    dict(point=(0.0,0.0,84.0),name="tm_base_cover"),
    dict(point=(0.0,0.0,92.0),name="top_stack"),
    dict(point=(0.0,0.0,101.0),name="tl_base_cover"),
    dict(point=(8.0,0.0,115.0),name="t_casette"),
    dict(point=(0.0,0.0,129.0),name="tt_base_cover"),
    dict(point=(0.0,0.0,137.0),name="top_plate")
]

aluminium_checking_tag = []

for part in aluminium_parts_point:
    x,y,z = part["point"]

    for dim,tag in all_volumes_tags:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(dim, tag)
        
        # Check if the point is inside this volume's bounding box
        if (xmin - tol <= x <= xmax + tol) and \
        (ymin - tol <= y <= ymax + tol) and \
        (zmin - tol <= z <= zmax + tol):
            
            aluminium_checking_tag.append(tag)

all_present = set(aluminim_tags).issubset(set(aluminium_checking_tag))

if not all_present:
    raise ValueError("Tag mis-match")

mesh_data = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3)
mesh_3d = mesh_data.mesh
gmsh.finalize()


# assign material propeties
V_mat = fem.functionspace(mesh_3d, ("DG",0))

rho = fem.Function(V_mat)
E = fem.Function(V_mat)
nu = fem.Function(V_mat)

# locate cells based on the physical group tag
cells_aluminium = mesh_data.cell_tags.find(1)
cells_steel = mesh_data.cell_tags.find(2)

# assign aluminium properties
rho.x.array[cells_aluminium] = 2.7e-9 
E.x.array[cells_aluminium] = 70000
nu.x.array[cells_aluminium] = 0.33

# assign steel properties
rho.x.array[cells_steel] = 7.8e-9
E.x.array[cells_steel] = 210000
nu.x.array[cells_steel] = 0.30  

E.x.scatter_forward()
nu.x.scatter_forward()
rho.x.scatter_forward()

# calculate Lame's parameter
lambda_ = (E * nu)/ ((1.0 + nu) * (1.0 - 2.0 * nu))
mu = E / (2.0 * (1 + nu))


# vector space for 3d displacement
V = fem.functionspace(mesh_3d, ("Lagrange",1, (mesh_3d.geometry.dim,)))

def epsilon(u):
    return ufl.sym(ufl.grad(u))

def sigma(u):
    return lambda_ * ufl.tr(epsilon(u)) * ufl.Identity(3) + 2 * mu * epsilon(u)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

k_form = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
m_form = rho * ufl.inner(u, v) * ufl.dx

# boundary condition
fdim = mesh_3d.topology.dim - 1
boundary_facets = mesh_data.facet_tags.find(10)
bc_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

# contrain those node to [0,0,0]
u_D = fem.Function(V)
u_D.x.array[:] = 0.0
bc = fem.dirichletbc(u_D, bc_dofs)

K = fem.petsc.assemble_matrix(fem.form(k_form), bcs=[bc],diag=1.0)
K.assemble()

M = fem.petsc.assemble_matrix(fem.form(m_form), bcs=[bc],diag=1.0)
M.assemble()

print("Setting up SLEPc solver...")

# initialize the eigen value problem
eps = SLEPc.EPS().create(mesh_3d.comm)

# pass the stiffness and mass matrics
eps.setOperators(K,M)

# Specify the mathematics: Generalized Hermitian
eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)

eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)

# Request 30 modes
num_modes = 30
eps.setDimensions(nev=num_modes)

eps.setTarget(0.0)
eps.setWhichEigenpairs(
    SLEPc.EPS.Which.TARGET_MAGNITUDE
)

st = eps.getST()
st.setType(SLEPc.ST.Type.SINVERT)
st.setShift(0.0)
eps.setFromOptions()
eps.solve()


# Check how many modes actually converged successfully
nconv = eps.getConverged()
print(f"Solver finished. Converged eigenpairs: {nconv}")

vr, vi = K.createVecRight(), K.createVecRight()
uh = fem.Function(V)

with io.XDMFFile(mesh_3d.comm, "modal_results.xdmf", "w") as xdmf:
    # First, write the 3D mesh geometry to the file
    xdmf.write_mesh(mesh_3d)
    
    # Loop through the requested number of modes (or converged modes, whichever is smaller)
    for i in range(min(nconv, num_modes)):

        # Extract the eigenvalue (lambda) and eigenvector (vr) from SLEPc
        eig_val = eps.getEigenpair(i, vr, vi)
        
        lambda_i = eig_val.real

        # Convert eigenvalue to physical frequency in Hertz
        freq_hz = np.sqrt(lambda_i) / (2 * np.pi)
        print(f"Mode {i+1}: Frequency = {freq_hz:.2f} Hz")
        
        # Copy the mathematical vector array into the FEniCSx physical displacement function
        uh.x.array[:] = vr.array
        
        # Synchronize data across processors (critical if running in parallel via MPI)
        uh.x.scatter_forward()
        
        # Name the data array so it is easily identifiable in ParaView
        uh.name = f"Mode_{i+1}_{freq_hz:.2f}Hz"
        
        # Write the mode shape to the XDMF file at "time step" i
        xdmf.write_function(uh, float(i))


