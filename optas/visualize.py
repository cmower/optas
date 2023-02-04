import os
import vtk
import casadi as cs
from vtkmodules.vtkFiltersSources import vtkCylinderSource, vtkSphereSource
from .spatialmath import *
from urdf_parser_py.urdf import Mesh, Cylinder, Sphere

class RobotVisualizer:

    def __init__(self, robot, q=None, show_global_link=True, alpha=1., show_link_names=False):

        self.init_robot(robot)

        # Create a rendering window and renderer
        self.ren = vtk.vtkRenderer()
        self.renWin = vtk.vtkRenderWindow()
        self.renWin.AddRenderer(self.ren)
        self.renWin.SetSize(int(1920*0.75), int(1080*0.75))
        self.camera = self.ren.GetActiveCamera()
        self.camera.SetPosition(2, 2, 2)
        self.camera.SetFocalPoint(0, 0, 0)
        self.camera.SetViewUp(0, 0, 1)

        # Create a renderwindowinteractor
        self.iren = vtk.vtkRenderWindowInteractor()
        self.iren.SetRenderWindow(self.renWin)

        style = vtk.vtkInteractorStyleTrackballCamera()
        self.iren.SetInteractorStyle(style)

        self.draw_grid_floor()
        if show_global_link:
            self.draw_link(I4())  # global link

        if q is None:
            q = [0.]*self.robot.ndof
        self.draw_robot(q, alpha)

        if show_link_names:
            for name, Tf in self.link_tf.items():

                tf = Tf(q)
                p = tf[:3, 3].toarray()

                actor = self.create_text_actor(name, p, scale=[2*0.001, 2*0.001, 2*0.001])
                self.ren.AddActor(actor)

    def init_robot(self, robot):

        # Setup robot attributes
        self.robot = robot
        self.urdf = robot.get_urdf()

        # Setup functions to compute visual origins in global frame
        q = cs.SX.sym('q', robot.ndof)
        self.link_tf = {}
        self.visual_tf = {}
        for link in self.urdf.links:

            name = link.name

            link_tf = robot.get_global_link_transform(link.name, q)
            self.link_tf[name] = cs.Function(f'link_tf_{name}', [q], [link_tf])

            xyz, rpy = robot.get_link_visual_origin(link)
            visl_tf = rt2tr(rpy2r(rpy), xyz)

            tf = link_tf @ visl_tf
            self.visual_tf[name] = cs.Function(f'visual_tf_{name}', [q], [tf])

    def _actor_from_stl(self, link, q):

        geometry = link.visual.geometry
        filename = geometry.filename

        if not os.path.exists(filename):
            relpath = self.robot.get_urdf_dirname()
            if relpath is None:
                raise RuntimeError(f"Unable to load mesh from filename: {filename}")
            filename = os.path.join(relpath, filename)

        reader = vtk.vtkSTLReader()
        reader.SetFileName(filename)

        if geometry.scale is not None:

            transform = vtk.vtkTransform()
            transform.Scale(*geometry.scale)

            transformFilter = vtk.vtkTransformPolyDataFilter()
            transformFilter.SetTransform(transform)
            transformFilter.SetInputConnection(reader.GetOutputPort())

            # Visualize the object
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(transformFilter.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)

        else:

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(reader.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)

        return actor

    def draw_robot(self, q, alpha):

        for link in self.urdf.links:

            geometry = link.visual.geometry
            tf = self.visual_tf[link.name](q)

            if isinstance(geometry, Mesh):

                if geometry.filename.lower().endswith('.stl'):
                    actor = self._actor_from_stl(link, q)
                else:
                    raise NotImplementedError(f"mesh file type for link '{link.name}' is not supported")

            elif isinstance(geometry, Sphere):

                sphere = vtkSphereSource()
                sphere.SetRadius(geometry.radius)
                sphere.SetThetaResolution(20)
                sphere.SetPhiResolution(20)

                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(sphere.GetOutputPort())

                actor = vtk.vtkActor()
                actor.SetMapper(mapper)

            elif isinstance(geometry, Cylinder):

                cylinder = vtkCylinderSource()
                cylinder.SetRadius(geometry.radius)
                cylinder.SetHeight(geometry.length)
                cylinder.SetResolution(20)

                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(cylinder.GetOutputPort())

                actor = vtk.vtkActor()
                actor.SetMapper(mapper)

                tf = tf @ r2t(rotx(0.5*cs.np.pi))

            else:
                raise NotImplementedError("link type is not supported")

            transform = vtk.vtkTransform()
            transform.SetMatrix(tf.toarray().flatten().tolist())
            actor.SetUserTransform(transform)

            actor.GetProperty().SetOpacity(alpha)

            self.ren.AddActor(actor)

    def create_text_actor(self, text, position, scale=[1, 1, 1], rgb=[0, 0, 0]):

        # Create a text source to generate the text
        textSource = vtk.vtkTextSource()
        textSource.SetText(text)
        textSource.Update()

        # Create a mapper to map the text source to graphics primitives
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(textSource.GetOutputPort())

        # Create an actor to represent the text
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Create a follower to position the text in 3D space
        follower = vtk.vtkFollower()
        follower.SetMapper(mapper)
        follower.SetPosition(*position)
        follower.SetScale(*scale)
        # follower.GetProperty().SetColor(*rgb)
        follower.GetProperty().SetOpacity(0.6)
        follower.SetCamera(self.camera)

        return follower

    @staticmethod
    def create_line_actor(start, end):

        points = vtk.vtkPoints()
        points.InsertNextPoint(*start)
        points.InsertNextPoint(*end)

        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, 0)
        line.GetPointIds().SetId(0, 1)

        lines = vtk.vtkCellArray()
        lines.InsertNextCell(line)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetLines(lines)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        return actor

    @staticmethod
    def create_cylinder_actor(start, end, radius):

        s = cs.np.array(start)
        e = cs.np.array(end)
        a = e - s
        l = cs.np.linalg.norm(a)
        p = 0.5*a
        y = a / l

        temp = cs.np.random.random(3)
        temp /= cs.np.linalg.norm(temp)

        x = cs.np.cross(y, temp)
        z = cs.np.cross(x, y)
        T = cs.np.eye(4)
        T[:3, 0] = x
        T[:3, 1] = y
        T[:3, 2] = z
        T[:3, 3] = p

        cylinder = vtkCylinderSource()
        cylinder.SetRadius(radius)
        cylinder.SetHeight(l)
        cylinder.SetResolution(20)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(cylinder.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        transform = vtk.vtkTransform()
        transform.SetMatrix(T.flatten().tolist())
        actor.SetUserTransform(transform)

        return actor

    def draw_grid_floor(self, ncells=10):
        assert ncells % 2 == 0, "ncells must be even!"
        nlines = ncells + 1

        offset = vtk.vtkTransform()
        offset.Translate(-0.5*float(ncells), -0.5*float(ncells), 0)

        for i in range(nlines):

            actor = self.create_line_actor([float(i), 0, 0], [float(i), float(ncells), 0])
            actor.SetUserTransform(offset)
            self.ren.AddActor(actor)

            actor = self.create_line_actor([0, float(i), 0], [float(ncells), float(i), 0])
            actor.SetUserTransform(offset)
            self.ren.AddActor(actor)

    def draw_link(self, T):

        scale = 0.1

        axis_scale = 1 * scale
        center_radius = 0.075 * scale
        axis_radius = 0.03 * scale

        # Extract data
        p = T[:3, 3].toarray().flatten()
        x = T[:3, 0].toarray().flatten()
        y = T[:3, 1].toarray().flatten()
        z = T[:3, 2].toarray().flatten()

        # Draw a sphere to represent link center
        center = vtkSphereSource()
        center.SetRadius(center_radius)
        center.SetThetaResolution(20)
        center.SetPhiResolution(20)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(center.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        transform = vtk.vtkTransform()
        transform.Translate(*p.tolist())
        actor.SetUserTransform(transform)
        self.ren.AddActor(actor)

        # Draw axes
        xaxis = self.create_cylinder_actor(p, p+axis_scale*x, axis_radius)
        xaxis.GetProperty().SetColor(1, 0, 0)
        self.ren.AddActor(xaxis)

        yaxis = self.create_cylinder_actor(p, p+axis_scale*y, axis_radius)
        yaxis.GetProperty().SetColor(0, 1, 0)
        self.ren.AddActor(yaxis)

        zaxis = self.create_cylinder_actor(p, p+axis_scale*z, axis_radius)
        zaxis.GetProperty().SetColor(0, 0, 1)
        self.ren.AddActor(zaxis)

    def start(self):
        # Enable user interface interactor
        self.iren.Initialize()
        self.renWin.Render()
        self.iren.Start()
