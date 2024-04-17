import simfempy
import matplotlib.pyplot as plt

#-------------------------------------------------------------
def get_fem_data(plotting=False, h=0.4, verbose=0):
    class FlowExample(simfempy.applications.navierstokes.Application):
        def defineGeometry(self, geom, _):
            points = [geom.add_point((-2, 0, 0), mesh_size=h),
                      geom.add_point((1, 0, 0), mesh_size=h),
                      geom.add_point((1, 1, 0), mesh_size=0.25 * h),
                      geom.add_point((2, 1, 0), mesh_size=0.25 * h),
                      geom.add_point((2, 0, 0), mesh_size=h),
                      geom.add_point((3, 0, 0), mesh_size=h),
                      geom.add_point((3, 2, 0), mesh_size=0.25 * h),
                      geom.add_point((4, 2, 0), mesh_size=0.25 * h),
                      geom.add_point((4, 0, 0), mesh_size=h),
                      geom.add_point((8, 0, 0), mesh_size=h),
                      geom.add_point((8, 3, 0), mesh_size=h),
                      geom.add_point((-2, 3, 0), mesh_size=h)]
            channel_lines = [geom.add_line(points[i], points[i + 1]) for i in range(-1, len(points) - 1)]
            channel_loop = geom.add_curve_loop(channel_lines)
            plane_surface = geom.add_plane_surface(channel_loop, holes=[])
            geom.add_physical([plane_surface], "Volume")
            geom.add_physical([channel_lines[0]], "Inflow")
            geom.add_physical([channel_lines[-2]], "Outflow")
            wall_lines = channel_lines[1:-2]
            wall_lines.append(channel_lines[-1])
            geom.add_physical(wall_lines, "Walls")

        def defineProblemData(self, problemdata):
            problemdata.bdrycond.set("Dirichlet", ["Walls", "Inflow"])
            problemdata.bdrycond.set("Neumann", "Outflow")
            problemdata.bdrycond.fct["Inflow"] = [lambda x, y, z: y * (3 - y), lambda x, y, z: 0]
            problemdata.params.scal_glob["mu"] = 1.
    flow_solver = simfempy.models.navierstokes.NavierStokes(application=FlowExample(), verbose=verbose)
    pp,u_flow = flow_solver.static()
    if plotting:
        data_flow = u_flow.tovisudata()
        flow_solver.application.plot(mesh=flow_solver.mesh, data=data_flow)
        plt.show()
    class HeatExample(simfempy.applications.application.Application):
        def defineProblemData(self, problemdata):
            problemdata.bdrycond.set("Dirichlet", ["Walls", "Inflow"])
            problemdata.bdrycond.set("Neumann", "Outflow")
            problemdata.bdrycond.fct["Inflow"] = lambda x, y, z: 290 + 30* max(0,y-0.5) * max(0,1.5 - y)
            problemdata.bdrycond.fct["Walls"] = lambda x, y, z: 290
            problemdata.params.scal_glob["kheat"] = 0.001
            problemdata.params.data["convection"] = u_flow.extract(name="v")
    heat_solver = simfempy.models.elliptic.Elliptic(mesh = flow_solver.mesh, application=HeatExample())
    result, u_heat = heat_solver.static(method="linear")
    data = u_heat.tovisudata()
    if plotting:
        heat_solver.application.plot(mesh=heat_solver.mesh, data=data)
        plt.show()
    return {'T':data['point']['U00'], 'p':heat_solver.mesh.points, 's':heat_solver.mesh.simplices, 'b':heat_solver.mesh.getBdryPoints("Inflow")}


#-------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    data = get_fem_data(plotting=True, verbose=1)
    T, points, bdry, s = data['T'], data['p'], data['b'], data['s']
    print(f"{T.shape=} {T.min()=} {T.max()=}")
    i2 = np.argsort(points[bdry,1])
    b = bdry[i2]
    plt.plot(points[b,1], T[b], '-X')
    plt.show()
    plt.tricontourf(points[:,0], points[:,1], s, T)
    plt.show()
